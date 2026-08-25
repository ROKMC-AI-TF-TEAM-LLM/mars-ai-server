# deploy_l40.md — L40 운영 서버 배포 런북

에어갭 내부망의 L40 48GB 단일 서버에 MARS를 배포하는 절차.
개발 노트북과의 차이는 §7 비교표 참조. 관련: roadmap.md 7단계(검증 항목).

---

## 0. 전제

- 서버: L40 48GB × 1, Linux x86_64, **Python 3.11.x**
- NVIDIA 드라이버: CUDA 12.x 호환 (vLLM 0.11.0 요구)
- 외부 네트워크 차단 (에어갭) — 모든 반입은 물리 매체/내부 저장소 경유
- Docker 불필요 (Milvus Lite는 pip 라이브러리)
- 🚨 **C 컴파일러(gcc) 필수** — 아래 참조

### 🚨 gcc가 없으면 vLLM이 죽는다

vLLM은 `torch.compile`/Triton으로 **런타임에 커널을 JIT 컴파일**한다. gcc가 없으면:

| 증상 | 시점 |
|---|---|
| `InductorError: Failed to find C compiler` | **기동 실패** |
| `--enforce-eager`를 주면 기동은 된다 | — |
| 그래도 `apply_grammar_bitmask` → 같은 오류 → **`EngineDeadError`** | **첫 `json_schema` 요청** |

두 번째가 특히 위험하다. 요청 하나가 실패하는 게 아니라 **엔진이 죽어 서버 전체가
못 쓰게 된다.** `response_format=json_schema`는 라우팅·검증의 **주 경로**이므로
(answer_rate 76%→91%의 근거) 사실상 모든 질의가 이 경로를 탄다.

```bash
# 확인
gcc --version || echo "설치 필요"

# RHEL 계열 (에어갭이면 RPM 반입 필요)
dnf install -y gcc
```

WSL AlmaLinux 9에서 실측: gcc 미설치 상태로 기동 → 실패, `--enforce-eager` →
기동은 되나 첫 구조화 요청에서 엔진 사망, `gcc 11.5.0` 설치 후 정상.

## 1. 반입물 준비 (인터넷 가능한 Linux 환경에서)

⚠ **wheel은 OS·아키텍처·파이썬 버전에 종속**된다. Windows 노트북에서 받은
wheel은 L40에서 안 맞는다 — 반드시 Linux x86_64 + Python 3.11에서 받을 것
(WSL 또는 `docker run -it python:3.11-slim bash` 활용).

⚠ `FlagEmbedding==1.3.3`은 **wheel이 없다** (1.4.0부터 제공). 순수 파이썬이라
sdist로 받아도 에어갭에서 설치되므로 `--no-binary FlagEmbedding`을 준다.
그 외에는 `--only-binary=:all:`로 sdist가 섞이지 않게 하는 것이 안전하다 —
에어갭에는 컴파일러가 없다.

🚨 **`pip download -r requirements.txt`는 실패한다** (`ResolutionImpossible`).
`vllm(transformers==4.57.1)`과 `FlagEmbedding(transformers==4.44.2)`이 정확히
못 박혀 있어 겹치는 버전이 없다 — venv를 나누는 이유가 다운로드에도 그대로 적용된다.
**venv별 파일로 따로 받는다.**

```bash
# ① 파이썬 패키지 (의존성 포함 전부) — venv 구성에 맞춰 두 번
docker run --rm -v "$PWD:/src:ro" -v "$PWD/wheels:/out" python:3.11-slim \
  bash -c "pip download -d /out -r /src/requirements-linux-app.txt --no-binary FlagEmbedding"
docker run --rm -v "$PWD:/src:ro" -v "$PWD/wheels:/out" python:3.11-slim \
  bash -c "pip download -d /out -r /src/requirements-linux-llm.txt"

pip download -d wheels-eval/ -r requirements-eval.txt  # 평가 쓸 때만

# ② 모델 3종 (HuggingFace에서 스냅샷)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("skt/A.X-4.0-Light", local_dir="models/A.X-4.0-Light")          # ~15GB (fp16 원본, GGUF 아님)
snapshot_download("BAAI/bge-m3", local_dir="models/bge-m3",
                  ignore_patterns=["onnx/*", "*.onnx", "imgs/*"])
snapshot_download("BAAI/bge-reranker-v2-m3", local_dir="models/bge-reranker-v2-m3")
EOF

# ③ 소스 코드: git bundle 또는 압축
git bundle create mars.bundle main
```

반입 목록: `wheels/`, `models/` 3종, `mars.bundle`(또는 소스 tar).

## 2. 설치 — venv를 반드시 2개로 분리

`requirements.txt`의 `vllm(transformers==4.57.1)`과
`FlagEmbedding(transformers<4.45)`은 **한 venv에 공존 불가** (실측 확인).
L40에서는 다음과 같이 나눈다:

| venv | 용도 | 설치 패키지 |
|---|---|---|
| `venv-llm` | vLLM 서빙 전용 | requirements.txt의 "서빙 코어" 블록 (vllm, torch, transformers, torchvision, torchaudio, tokenizers, triton) |
| `venv-app` | MARS 앱 + 임베딩/리랭커 서버 | 나머지 전부 + FlagEmbedding (transformers는 FlagEmbedding이 맞는 버전을 끌고 옴) |

요구사항 파일이 venv별로 나뉘어 있으므로 그대로 쓴다 (패키지를 손으로 나열하지 않는다).

```bash
tar xf mars.tar && cd mars-ai-server   # 또는 git clone mars.bundle

python3.11 -m venv venv-llm
venv-llm/bin/pip install --no-index --find-links wheels/ -r requirements-linux-llm.txt

python3.11 -m venv venv-app
venv-app/bin/pip install --no-index --find-links wheels/ -r requirements-linux-app.txt

# 검증: 둘 다 통과해야 한다
venv-llm/bin/pip check && venv-app/bin/pip check
```

> `setuptools==75.6.0`은 `requirements-linux-app.txt` 맨 앞에 있어 따로 깔 필요가 없다.
> 빠뜨리면 최신 setuptools가 깔려 `pymilvus`가 요구하는 `pkg_resources`가 없어 터진다.
>
> `triton`은 **3.4.0**이다. `requirements.txt`에 적힌 3.5.0은 `torch 2.8.0`이
> `triton==3.4.0`을 정확히 못 박아 **설치 불가**다 (Windows에는 triton이 없어
> 개발에서 드러나지 않는다).

모델은 프로젝트의 `models/` 아래(또는 임의 경로)에 배치한다.

## 3. .env 작성 (.env.example 기준)

개발용 `.env.dev.example`이 아니라 **`.env.example`을 복사**해서 수정:

```bash
cp .env.example .env
```

L40에서 반드시 확인할 항목:

```bash
AX_MODEL_NAME=/srv/mars/models/A.X-4.0-Light   # vLLM serve 경로와 동일하게 (Hub ID 금지)
EMBEDDING_DEVICE=cuda                           # 개발은 cpu였음
RERANKER_DEVICE=cuda
EMBEDDING_MODEL_PATH=/srv/mars/models/bge-m3
RERANKER_MODEL_PATH=/srv/mars/models/bge-reranker-v2-m3
MILVUS_LITE_PATH=./data/milvus_ax.db            # ★ 파일 경로 (개발의 http://... 아님)
LOG_LEVEL=INFO                                  # 개발은 DEBUG
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`serving/start_vllm.sh`의 모델 경로도 실제 반입 경로로 수정한다.

## 4. 기동 (순서대로, 각 프로세스 독립 실행)

```bash
# ① vLLM :8000 — venv-llm (기동 수 분 소요, 약 37GB VRAM 선점)
source venv-llm/bin/activate && bash serving/start_vllm.sh

# ② 임베딩 :8001 — venv-app
PYTHONPATH=src venv-app/bin/python serving/embedding_server.py

# ③ 리랭커 :8002 — venv-app
PYTHONPATH=src venv-app/bin/python serving/reranker_server.py

# ④ 문서 적재 (최초 1회 / 갱신 시)
PYTHONPATH=src venv-app/bin/python scripts/bulk_ingest.py \
    --dir /srv/mars/docs_in --domain GENERAL --department HQ --visibility ALL

# ⑤ MARS API :9000 — venv-app, ★ 단일 워커 강제 (--workers 금지, Milvus Lite 파일 락)
PYTHONPATH=src venv-app/bin/python -m uvicorn main:app --host 0.0.0.0 --port 9000
```

상시 운영은 systemd 유닛 4개(또는 tmux/supervisor)로 감싸는 것을 권장.
재기동 순서는 항상 vLLM → 임베딩 → 리랭커 → main.py.

VRAM 예산: vLLM 0.78×48≈37GB + BGE-M3 1~2GB + 리랭커 1.6GB ≈ 41GB (여유 ~7GB).

## 5. 배포 검증 체크리스트 (roadmap 7단계)

```bash
curl localhost:8000/v1/models        # vLLM 기동 확인
curl localhost:8001/health           # 임베딩
curl localhost:8002/health           # 리랭커
curl localhost:9000/health           # MARS
curl localhost:9000/documents        # 적재 인벤토리

make test-all                        # 통합 테스트 14개 (venv-app에 pytest 필요 시 wheels로 설치)
```

- [ ] **아웃바운드 0건 실측**: 네트워크 차단 상태에서 전체 스택 기동 성공 확인
      (모델 경로가 없으면 즉시 실패해야 정상 — Hub 폴백 없음)
- [ ] **tool-calling 성공률**: 라우터/검증의 1차 성공률 측정.
      로그에서 `tool-call 파싱 실패` WARNING 빈도로 확인 (3단 폴백 발동률)
- [ ] Milvus Lite가 HNSW 인덱스와 query_iterator를 지원하는지 확인
      (미지원 시: 인덱스 타입 조정 / iterator 폴백 경고 로그 확인)
- [ ] `vllm bench serve`로 동시성 파라미터(max_num_seqs) 확정
- [ ] chars_per_token=2.2 근사를 실제 군 문서로 보정 (config.CHARS_PER_TOKEN)
- [ ] SSE E2E: `curl -N`으로 status → text → sources → done 순서 확인
- [ ] 감사 로그 기록 확인 (`data/audit_log.jsonl`)
- [ ] ACL E2E: DEPT_ONLY 문서를 타 부서 계정으로 질의 → 미노출 확인

## 6. 운영 중 문서 갱신

```bash
PYTHONPATH=src venv-app/bin/python scripts/reindex_document.py \
    --file 수정된문서.pdf --domain GENERAL --department HQ
```
서버 재시작 불필요 — BM25 캐시가 빌드 버전(uuid)으로 갱신을 자동 감지한다.
BM25 전체 재빌드가 동반되므로 야간 배치를 권장.

## 7. 개발 노트북 vs L40 비교

| | 개발 노트북 (Windows) | L40 (운영) |
|---|---|---|
| LLM | llama.cpp + GGUF Q4 (`tools/`) | **vLLM + 원본 fp16** (`serving/start_vllm.sh`) |
| venv | 1개 (vllm 미설치) | **2개 분리** (venv-llm / venv-app) |
| 벡터DB | Docker Milvus standalone :19530 | **Milvus Lite 파일** (`./data/milvus_ax.db`), Docker 불필요 |
| 디바이스 | 임베딩/리랭커 cpu | cuda |
| .env | `.env.dev.example` | `.env.example` 기준 |
| 네트워크 | 인터넷 (다운로드 가능) | 에어갭 (wheel/모델 반입) |
| 로그 | DEBUG | INFO |
| tool-calling | 잠정 통과 | **최종 검증 지점** |
| dev_setup.ps1 / tools/ | 사용 | 사용 안 함 |

---

## 8. 개발 노트북(Windows) 에어갭 반입

내부망 개발 노트북에도 같은 환경을 만들어야 할 때. **L40과 wheel을 공유할 수 없다**
(OS·아키텍처 종속).

### 8-1. 요구사항 파일

`requirements.txt`를 쓰지 않는다. **`requirements-dev-windows.lock`** 을 쓴다:

- `vllm` 계열 제외 — Windows 미지원 (llama.cpp로 서빙)
- `milvus-lite` 제외 — Windows 미지원 (Docker Milvus standalone)
- `transformers`는 **4.44.2** (FlagEmbedding이 결정). 운영의 4.57.1은 vllm 요구 버전
- `torch`는 **CPU 빌드**(`2.8.0+cpu`) — 개발은 임베딩·리랭커를 CPU로 돌린다

`.lock`은 테스트가 통과한 환경을 그대로 굳힌 것이다. 하위 의존성까지 고정돼 있어
내부망에서 다른 버전이 깔릴 여지가 없다.

### 8-2. wheel 받기 (인터넷 되는 Windows, 같은 Python 3.11)

```powershell
python -m pip download -d wheels-win --no-binary FlagEmbedding -r requirements-dev-windows.lock
```

받은 뒤 sdist 확인 — `FlagEmbedding`·`kiwipiepy_model` 둘만 나와야 정상이다
(각각 순수 파이썬 / 사전 데이터라 컴파일 불필요):

```powershell
Get-ChildItem wheels-win | Where-Object { $_.Name -notlike "*.whl" }
```

### 8-3. 반입 용량 줄이기 — 공용 wheel 분리 (선택)

`py3-none-any.whl`은 OS를 안 가리므로 Windows·Linux가 같은 파일을 쓴다.
반입 승인 절차가 번거로우면 셋으로 나눈다 (실측: **55개 / 107MB** 절감,
`kiwipiepy_model` 76MB가 대부분):

```
wheels-shared/   양쪽 공용 (py3-none-any)
wheels-win/      Windows 전용 (win_amd64)
wheels-linux/    Linux 전용 (manylinux)
```

`--find-links`를 여러 번 주면 되므로 설치 명령은 한 줄만 늘어난다.
목록은 `scripts/wheel_list*.csv` 참조.

### 8-4. Milvus 반입 — Docker 이미지 1개

Windows는 `milvus-lite`를 못 쓰므로 **Docker standalone**(etcd 내장)으로 띄운다.
별도 etcd·minio 컨테이너가 필요 없어 **이미지 하나면 된다**.

```powershell
# 인터넷 되는 곳에서
docker pull milvusdb/milvus:v2.5.4
docker save milvusdb/milvus:v2.5.4 -o milvus-v2.5.4.tar    # 약 2.4GB

# 내부망에서
docker load -i milvus-v2.5.4.tar
```

설정 파일 2개(`serving/milvus-dev/embedEtcd.yaml`, `user.yaml`)는 저장소에 있으므로
소스와 함께 들어간다. 기동은 `scripts/dev_setup.ps1`의 [5/5] 단계와 동일하다.

### 8-5. Windows 반입 목록

| 항목 | 크기 | 비고 |
|---|---|---|
| Python 3.11.x 설치 파일 | ~30MB | 내부망에 없을 수 있다 |
| wheel 묶음 | ~0.5GB | `requirements-dev-windows.lock` 기준 |
| **Docker Desktop 설치 파일** | ~500MB | Milvus 구동에 필수 |
| **Milvus 이미지 tar** | ~2.4GB | `docker save` |
| `A.X-4.0-Light-Q4_K_M.gguf` | ~4.4GB | llama.cpp용 |
| llama.cpp 릴리스 + cudart | ~50MB | `tools/llama.cpp/` |
| `bge-m3`, `bge-reranker-v2-m3` | ~4GB | 임베딩·리랭커 |
| 소스 (`git bundle`) | 수 MB | |

### 8-6. 설치·검증

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels-win `
    -r requirements-dev-windows.lock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`--no-index`가 핵심이다 — PyPI를 아예 보지 않으므로, 이게 통과하면 에어갭에서도 된다.

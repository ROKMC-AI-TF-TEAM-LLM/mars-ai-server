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

**gcc만으로는 부족하다 — Python 헤더도 필요하다.** Triton이 `cuda_utils.c`를
컴파일할 때 `-I/usr/include/python3.11`을 쓰므로 `Python.h`가 있어야 한다.
없으면 gcc가 있어도 컴파일이 실패하고 결과는 같다 (`EngineDeadError`).

```bash
# 확인
gcc --version                     || echo "gcc 설치 필요"
ls /usr/include/python3.11/Python.h || echo "python3.11-devel 설치 필요"

# RHEL 계열 (에어갭이면 RPM 반입 필요)
dnf install -y gcc python3.11-devel
```

**반입 목록에 RPM을 넣을 것**: `gcc`, `python3.11-devel`과 그 의존성
(`cpp`, `binutils`, `glibc-devel`, `kernel-headers`, `libxcrypt-devel`, `make` 등).
인터넷 되는 같은 버전 RHEL/AlmaLinux에서 `dnf download --resolve gcc python3.11-devel`로 모은다.

WSL AlmaLinux 9.8 실측 단계별 증상:

| 상태 | 결과 |
|---|---|
| gcc 없음 | 기동 실패 (`InductorError`) |
| gcc 없음 + `--enforce-eager` | 기동은 됨 → **첫 구조화 요청에서 엔진 사망** |
| gcc 있음, 헤더 없음 | 기동됨 → 요청 시 `subprocess.CalledProcessError` → 엔진 사망 |
| gcc + python3.11-devel | 정상 |

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

```bash
# ④ OS 패키지 (RPM) — 없으면 vLLM이 죽는다. §0 참조
#    같은 버전의 RHEL/AlmaLinux에서 받아야 한다
dnf download --resolve --destdir rpms/ gcc gcc-c++ python3.11-devel
```

### 1-1. L40 반입 목록 (체크리스트)

| ☐ | 항목 | 크기(약) | 없으면 생기는 일 |
|---|---|---:|---|
| ☐ | `wheels/` (app + llm 두 벌) | ~5 GB | 설치 불가 |
| ☐ | **`rpms/` (gcc, python3.11-devel + 의존성)** | ~50 MB | **첫 질의에 엔진 사망** ★ |
| ☐ | `models/A.X-4.0-Light` (fp16 원본) | ~15 GB | LLM 기동 불가 |
| ☐ | `models/bge-m3` | ~2.3 GB | 임베딩 불가 |
| ☐ | `models/bge-reranker-v2-m3` | ~2.3 GB | 리랭크 불가 |
| ☐ | `mars.bundle` (소스) | 수 MB | — |
| ☐ | Python 3.11.x (OS에 없으면) | ~30 MB | 전부 불가 |

**합계 약 25 GB.** Docker 이미지는 **필요 없다** (Milvus Lite는 pip 라이브러리).

> ★ 표시가 가장 위험하다. RPM을 빠뜨려도 **설치와 기동은 성공하고**,
> 실사용 첫 질의에서 엔진이 죽는다. §0과 `docs/troubleshooting.md` ⑤⑥ 참조.

### 1-2. 반입 전 검증 (인터넷 되는 곳에서)

물리 반입은 되돌리기 비싸므로, **떠나기 전에** 확인한다.

```bash
# wheel 목록에 setuptools가 있는가 (pip freeze는 기본 제외한다)
ls wheels/ | grep -i setuptools || echo "★ setuptools 누락 — pymilvus가 터진다"

# triton이 3.4.0인가 (3.5.0은 torch 2.8.0과 설치 불가)
ls wheels/ | grep -i triton

# 에어갭 설치 예행연습: --no-index로 PyPI를 아예 차단하고 깔아 본다
docker run --rm -v "$PWD:/src:ro" python:3.11-slim bash -c \
  "python -m venv /tmp/v && /tmp/v/bin/pip install --no-index \
   --find-links /src/wheels -r /src/requirements-linux-app.txt && /tmp/v/bin/pip check"
```

마지막 명령이 통과하면 에어갭에서도 통과한다. **`--no-index`가 핵심이다.**

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

## 4. 기동 매뉴얼

### 4-0. 순서를 지켜야 하는 이유

```
① vLLM(8000) → ② 임베딩(8001) → ③ 리랭커(8002) → ④ 적재 → ⑤ API(9000)
```

**vLLM을 반드시 먼저** 올린다. vLLM은 VRAM을 큰 연속 블록으로 선점하므로,
임베딩·리랭커가 먼저 자리를 잡으면 파편화로 기동이 실패한다.
재기동할 때도 같은 순서다.

**띄우기 전에 이미 떠 있는지 확인한다** — 중복 기동은 VRAM 부족으로 실패하는데,
오류 메시지가 "메모리 부족"이라 원인을 착각하기 쉽다.

```bash
curl -s localhost:8000/v1/models >/dev/null && echo "이미 떠 있음" || echo "비어 있음"
```

### 4-1. 기동

```bash
cd /srv/mars

# ① vLLM :8000 — venv-llm (기동 수 분 소요, 약 37GB VRAM 선점)
setsid bash serving/start_vllm.sh > /var/log/mars/vllm.log 2>&1 < /dev/null &

# ② 임베딩 :8001 — venv-app
PYTHONPATH=src setsid venv-app/bin/python serving/embedding_server.py \
    > /var/log/mars/embedding.log 2>&1 < /dev/null &

# ③ 리랭커 :8002 — venv-app
PYTHONPATH=src setsid venv-app/bin/python serving/reranker_server.py \
    > /var/log/mars/reranker.log 2>&1 < /dev/null &

# ④ 문서 적재 (최초 1회 / 갱신 시) — ★ 도메인은 문서별로 지정한다
PYTHONPATH=src venv-app/bin/python scripts/bulk_ingest.py \
    --dir /srv/mars/docs_in/훈령 --domain DIRECTIVE --department HQ --visibility ALL

# ⑤ MARS API :9000 — venv-app, ★ 단일 워커 강제 (--workers 금지, Milvus Lite 파일 락)
PYTHONPATH=src setsid venv-app/bin/python -m uvicorn main:app \
    --host 0.0.0.0 --port 9000 > /var/log/mars/api.log 2>&1 < /dev/null &
```

> `setsid` 와 `< /dev/null` 이 필요하다. `&` 만 쓰면 **터미널을 닫을 때 함께 죽는다.**
> systemd로 감싸면 이 문제가 사라지므로 상시 운영은 systemd 유닛 4개를 권장한다.

> ★ **`--domain`은 디렉터리마다 따로 준다.** 전체를 한 번에 한 도메인으로 적재하면
> 훈령이 `HR`로 들어가는 식이 되어 도메인 한정 검색에서 배제된다
> (실측: `hit@fuse` 100% → 88.5%, `docs/troubleshooting.md` ⑭).

### 4-2. 기동 확인

```bash
until curl -s -m 3 localhost:8000/v1/models >/dev/null; do sleep 10; done; echo ":8000 OK"
for p in 8001 8002 9000; do
  until curl -s -m 3 "localhost:$p/health" >/dev/null; do sleep 5; done; echo ":$p OK"
done
```

**여기까지 통과해도 배포가 끝난 게 아니다.** §5의 `json_schema` 검증을 반드시 거친다.

### 4-3. 정지

```bash
pkill -f "bin/vllm"          # ⚠️ pkill -f "vllm serve"는 자기 자신을 죽인다
pkill -f embedding_server
pkill -f reranker_server
pkill -f "uvicorn main:app"
```

### 4-4. VRAM 예산

| 프로세스 | VRAM |
|---|---:|
| vLLM (0.78 × 48GB) | ~37 GB |
| BGE-M3 임베딩 | 1~2 GB |
| bge-reranker-v2-m3 | ~1.6 GB |
| **합계** | **~41 GB** (여유 ~7 GB) |

개발 노트북(6GB)과 달리 L40은 여유가 있어 임베딩·리랭커를 **`cuda`로 둔다.**

## 5. 배포 검증 체크리스트 (roadmap 7단계)

```bash
curl localhost:8000/v1/models        # vLLM 기동 확인
curl localhost:8001/health           # 임베딩
curl localhost:8002/health           # 리랭커
curl localhost:9000/health           # MARS
curl localhost:9000/documents        # 적재 인벤토리

make test-all                        # 통합 테스트 14개 (venv-app에 pytest 필요 시 wheels로 설치)
```

### 🚨 헬스체크만으로 끝내지 않는다

`python3.11-devel`이 없으면 **위 5개가 전부 통과하고 첫 실사용 질의에서 엔진이 죽는다.**
구조화 출력은 기동 시점에 컴파일하지 않기 때문이다. 아래를 반드시 실행한다.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$AX_MODEL_NAME"'","messages":[{"role":"user","content":"테스트"}],
       "response_format":{"type":"json_schema","json_schema":{"name":"t","schema":
       {"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}}'
```

**이 요청이 성공해야 배포가 끝난 것이다.** 실패하거나 엔진이 죽으면 §0으로 돌아간다.

- [ ] **`json_schema` 요청 1건 성공** (위 명령) ★ 가장 중요
- [ ] **아웃바운드 0건 실측**: 네트워크 차단 상태에서 전체 스택 기동 성공 확인
      (모델 경로가 없으면 즉시 실패해야 정상 — Hub 폴백 없음)
- [ ] **tool-calling 성공률**: 라우터/검증의 1차 성공률 측정.
      로그에서 `tool-call 파싱 실패` WARNING 빈도로 확인 (3단 폴백 발동률)
- [ ] 기동 로그에 `Milvus Lite(임베디드) 모드로 접속한다`가 찍히는지 확인
      (`서버 모드`가 찍히면 `.env`의 `MILVUS_LITE_PATH`가 잘못됐다)
- [ ] `query_iterator` 폴백 경고가 없는지 확인 (있으면 16,384행까지만 조회된다)
- [ ] 문서별 도메인이 올바른지 (`scripts/eval_retrieval.py` — `hit@fuse` 확인)
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

### 8-4. Milvus 반입·구동 (Windows) — Docker 이미지 1개

Windows에는 `milvus-lite` 휠이 없다. **Docker standalone**(etcd 내장)으로 띄운다.
별도 etcd·MinIO 컨테이너가 필요 없어 **이미지 하나면 된다**.

> 코드는 `MILVUS_LITE_PATH` 값의 형태로 두 모드를 자동 판별한다
> ([vectorstore.py](../src/ax_rag/shared/vectorstore.py)의 `get_client`).
> 값이 `http://...` 면 서버 모드, 파일 경로면 Lite 모드다.
> Windows에서 파일 경로를 주면 **해법을 알려주는 오류**와 함께 즉시 멈춘다.

#### ① 반입

```powershell
# 인터넷 되는 곳에서 — 태그를 반드시 고정한다 (pymilvus 2.5.4와 짝)
docker pull milvusdb/milvus:v2.5.4
docker save milvusdb/milvus:v2.5.4 -o milvus-v2.5.4.tar    # 약 2.4GB

# 내부망에서
docker load -i milvus-v2.5.4.tar
docker images milvusdb/milvus                              # 태그 확인
```

**Docker Desktop 설치 파일(~500MB)도 함께 반입해야 한다.** 내부망 Windows에
Docker가 없으면 Milvus를 띄울 방법이 없다.

#### ② 기동

설정 파일 2개(`embedEtcd.yaml`, `user.yaml`)는 저장소에 있으므로 소스와 함께 들어온다.

```powershell
docker compose -f serving/milvus-dev/docker-compose.yml up -d
docker compose -f serving/milvus-dev/docker-compose.yml ps
```

`scripts/dev_setup.ps1`의 [5/5] 단계도 **같은 컨테이너**(`ax-milvus-dev`)를 만든다.
이미지·볼륨이 같아 둘을 섞어 써도 데이터가 갈리지 않는다.

#### ③ .env 설정

```powershell
MILVUS_LITE_PATH=http://localhost:19530     # ★ 파일 경로가 아니다
EMBEDDING_DEVICE=cpu                        # 노트북 GPU는 LLM이 쓴다
RERANKER_DEVICE=cpu
```

`.env.dev.example`이 이 값들로 되어 있으므로 그대로 복사하면 된다.

#### ④ 확인

```powershell
curl http://localhost:9091/healthz          # Milvus 자체 헬스체크
.\.venv\Scripts\python.exe -c "from pymilvus import MilvusClient; `
    print(MilvusClient('http://localhost:19530').list_collections())"
```

기동 로그에 `Milvus 서버 모드로 접속한다`가 찍히면 정상이다.

#### ⑤ 정지·정리

```powershell
docker compose -f serving/milvus-dev/docker-compose.yml down       # 정지 (데이터 유지)
docker compose -f serving/milvus-dev/docker-compose.yml down -v    # 데이터까지 삭제
```

#### ⚠️ Docker Milvus로 검증한 것은 운영 검증이 아니다

**서버 모드와 Lite는 동작이 다르다.** 실제로 두 건이 운영에서만 터졌다:

| | 서버 (Windows 개발) | Lite (운영 L40) |
|---|---|---|
| HNSW 인덱스 | 수용 | **거부** → 컬렉션 생성 실패 |
| search 결과 PK | `hit["chunk_id"]` | `hit["id"]` → **`KeyError`** |

둘 다 코드에서 해결했지만(`AUTOINDEX`, `_primary_key`), **같은 종류의 차이가 또
있을 수 있다.** Windows Docker Milvus는 로직·프롬프트 개발용으로만 쓰고,
**배포 전 최종 검증은 반드시 Lite 환경(WSL 또는 L40)에서** 한다.
상세는 `docs/troubleshooting.md` ⑩⑪⑫.

#### 앱만 Windows에 두고 Milvus를 WSL에 둘 수는 없다

Milvus Lite는 서버가 아니라 **임베디드 라이브러리**다. **유닉스 도메인 소켓**으로
통신하므로 포워딩할 TCP 포트가 없고, 동봉 바이너리가 리눅스 ELF다.
vLLM·임베딩·리랭커는 TCP라 WSL 분리가 가능하지만 **Milvus Lite는 불가능하다.**

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

# deploy_l40.md — L40 운영 서버 배포 런북

에어갭 내부망의 L40 48GB 단일 서버에 MARS를 배포하는 절차.
개발 노트북과의 차이는 §7 비교표 참조. 관련: roadmap.md 7단계(검증 항목).

---

## 0. 전제

- 서버: L40 48GB × 1, Linux x86_64, **Python 3.11.x**
- NVIDIA 드라이버: CUDA 12.x 호환 (vLLM 0.11.0 요구)
- 외부 네트워크 차단 (에어갭) — 모든 반입은 물리 매체/내부 저장소 경유
- Docker 불필요 (Milvus Lite는 pip 라이브러리), root 불필요

## 1. 반입물 준비 (인터넷 가능한 Linux 환경에서)

⚠ **wheel은 OS·아키텍처·파이썬 버전에 종속**된다. Windows 노트북에서 받은
wheel은 L40에서 안 맞는다 — 반드시 Linux x86_64 + Python 3.11에서 받을 것
(WSL 또는 `docker run -it python:3.11-slim bash` 활용).

```bash
# ① 파이썬 패키지 (의존성 포함 전부)
pip download -d wheels/ -r requirements.txt
pip download -d wheels/ "setuptools==75.6.0"          # pymilvus의 pkg_resources 요구
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

```bash
tar xf mars.tar && cd mars-ai-server   # 또는 git clone mars.bundle

# venv-llm
python3.11 -m venv venv-llm
venv-llm/bin/pip install --no-index --find-links wheels/ \
    vllm==0.11.0 torch==2.8.0 transformers==4.57.1 torchvision==0.23.0 \
    torchaudio==2.8.0 tokenizers==0.22.1 triton==3.5.0

# venv-app
python3.11 -m venv venv-app
venv-app/bin/pip install --no-index --find-links wheels/ setuptools==75.6.0
venv-app/bin/pip install --no-index --find-links wheels/ \
    langgraph==0.2.62 langchain-core==0.3.29 langchain-openai==0.2.14 \
    langchain-text-splitters==0.3.4 pymilvus==2.5.4 milvus-lite==2.4.11 \
    FlagEmbedding==1.3.3 kiwipiepy==0.22.2 bm25s==0.2.5 pdfplumber==0.11.10 \
    fastapi==0.115.6 "uvicorn[standard]==0.34.0" pydantic==2.10.4 \
    python-dotenv==1.0.1 requests==2.32.3
```

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

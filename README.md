# mars-ai-server — MARS: 군 문서 RAG + 멀티 에이전트

군 내부 문서(법령·훈령·규정) 검색 챗봇 **MARS**(Marine Artificial intelligence
Retrieval System). 온프레미스, 에어갭 내부망.
LangGraph + vLLM(A.X 4.0 Light) + Milvus Lite + BGE-M3 + bge-reranker-v2-m3 + bm25s/Kiwi.

- 설계: [docs/architecture.md](docs/architecture.md)
- 구현 스펙 (우선 문서): [docs/interfaces.md](docs/interfaces.md)
- 작업 순서: [docs/roadmap.md](docs/roadmap.md)
- 구현 규칙: [CLAUDE.md](CLAUDE.md)

## 개발 환경 준비

```bash
python -m venv .venv          # Python 3.11 필수
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install "setuptools==75.6.0"  # 주의: pymilvus 2.5.4는 pkg_resources 필요 (setuptools 81+에서 제거됨)
pip install -e ".[dev]"       # 개발 도구 (pytest, ruff)
pip install -r requirements.txt   # 런타임 의존성 (L40 서버 기준)
cp .env.example .env
```

## 명령

```bash
make test          # 유닛 테스트만
make test-all      # 통합 테스트 포함 (로컬 서비스 필요)
make lint          # ruff check
make format        # ruff format
langgraph dev      # 그래프 시각화 디버깅 (개발 노트북 전용)
```

## 개발 환경 재구축 (초기화되는 PC용, clone부터)

### 전원 끄기 전 (백업)

```powershell
git push                                        # ★ 필수: push 안 한 커밋은 초기화 시 소멸
Copy-Item models, tools -Destination D:\backup\ -Recurse   # 선택: 모델 9GB 재다운로드 방지
```

### 재부팅 후 — 최초 1회 셋업

```powershell
# 0) 시스템 요구: Python 3.11.x, git, Docker Desktop (설치 후 실행 상태)
git clone <저장소 URL> mars-ai-server
cd mars-ai-server
# ★ -ExecutionPolicy Bypass 필수: Windows 기본 정책(Restricted)에서는 .ps1 실행이 차단된다
powershell -ExecutionPolicy Bypass -File scripts\dev_setup.ps1          # venv+의존성, .env, 모델(~9GB), llama.cpp, Milvus 컨테이너
# 모델 백업을 복원한 경우: models/, tools/ 붙여넣은 뒤
powershell -ExecutionPolicy Bypass -File scripts\dev_setup.ps1 -SkipModels
# Docker Desktop을 아직 설치 못 했다면: -SkipDocker로 나머지만 먼저 셋업
```

### 서버 기동 (순서대로, 각각 별도 터미널)

```powershell
docker start ax-milvus-dev                                                        # 벡터DB :19530 (셋업 직후엔 이미 실행 중)
powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1                # 1) LLM :8000
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe serving\embedding_server.py     # 2) 임베딩 :8001
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe serving\reranker_server.py      # 3) 리랭커 :8002
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 9000   # 4) API :9000
```

### 문서 적재 (초기화 시 벡터DB가 비므로 재적재 필수, 2·3번 서버 기동 후)

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\bulk_ingest.py --dir sample_docs --domain HR --department HR_TEAM --visibility ALL
# 문서 갱신 시:
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\reindex_document.py --file <파일> --domain <도메인> --department <부서>
```

### 동작 확인

```powershell
.\.venv\Scripts\python.exe -m pytest -q         # 유닛 테스트
curl http://localhost:9000/health               # API 헬스체크
curl -N -X POST http://localhost:9000/query -H "Content-Type: application/json" --data-binary "@질문.json"   # SSE 확인
```

- 초기화로 사라지는 것: venv, .env, 모델/도구(백업 가능), Milvus 적재 데이터,
  BM25 인덱스, 감사 로그. 코드·docs는 push했다면 clone으로 복구된다.

## 개발 노트북에서 LLM 띄우기 (선택)

Windows에서는 vLLM 실행이 불가하므로, 프롬프트/로직 검증용으로 llama.cpp를 쓴다:

1. `models/A.X-4.0-Light-Q4_K_M.gguf` (GGUF Q4_K_M) 준비
2. llama.cpp 릴리스 바이너리를 `tools/llama.cpp/`에 압축 해제
3. `powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1` → 8000 포트, `.env` 수정 불필요

tool-calling 동작은 vLLM 파서와 다를 수 있으므로 노트북 통과는 잠정 통과로
취급한다 (roadmap.md 4단계 주의사항).

## 실행 (L40 서버)

네 개의 독립 프로세스를 같은 서버에서 기동한다 (architecture.md §2):

1. vLLM (8000): `serving/start_vllm.sh`
2. 임베딩 서버 (8001): `python serving/embedding_server.py`
3. 리랭커 서버 (8002): `python serving/reranker_server.py`
4. 본 서버 (9000): `uvicorn main:app --port 9000` — **단일 워커 강제** (Milvus Lite 파일 락)

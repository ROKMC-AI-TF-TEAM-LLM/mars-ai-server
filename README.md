# mars-ai-server — A.X 내부 문서 RAG + 멀티 에이전트

사내 업무 문서 검색 챗봇 (온프레미스, 에어갭 내부망).
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

```powershell
# 0) 시스템 요구: Python 3.11.x, git, Docker Desktop (설치 후 실행 상태)
git clone <저장소 URL> mars-ai-server
cd mars-ai-server
powershell -File scripts\dev_setup.ps1     # venv+의존성, .env, 모델(~9GB), llama.cpp, Milvus 컨테이너
```

이후 기동 순서는 스크립트 마지막 출력 참고 (LLM → 임베딩 → 리랭커 → 적재 → main.py).

- 모델 재다운로드(~9GB)를 피하려면 초기화 전에 `models/`와 `tools/` 폴더를
  외장/비초기화 드라이브에 백업하고, 복원 후 `-SkipModels`로 실행:
  `powershell -File scripts\dev_setup.ps1 -SkipModels`
- Milvus 적재 데이터(`data/milvus-docker/`)와 BM25 인덱스는 초기화되면
  사라지므로 재적재 필요 (4번 단계). 감사 로그도 새로 시작된다.

## 개발 노트북에서 LLM 띄우기 (선택)

Windows에서는 vLLM 실행이 불가하므로, 프롬프트/로직 검증용으로 llama.cpp를 쓴다:

1. `models/A.X-4.0-Light-Q4_K_M.gguf` (GGUF Q4_K_M) 준비
2. llama.cpp 릴리스 바이너리를 `tools/llama.cpp/`에 압축 해제
3. `powershell -File serving\start_llm_dev.ps1` → 8000 포트, `.env` 수정 불필요

tool-calling 동작은 vLLM 파서와 다를 수 있으므로 노트북 통과는 잠정 통과로
취급한다 (roadmap.md 4단계 주의사항).

## 실행 (L40 서버)

네 개의 독립 프로세스를 같은 서버에서 기동한다 (architecture.md §2):

1. vLLM (8000): `serving/start_vllm.sh`
2. 임베딩 서버 (8001): `python serving/embedding_server.py`
3. 리랭커 서버 (8002): `python serving/reranker_server.py`
4. 본 서버 (9000): `uvicorn main:app --port 9000` — **단일 워커 강제** (Milvus Lite 파일 락)

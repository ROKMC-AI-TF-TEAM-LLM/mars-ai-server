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

## 실행 (L40 서버)

네 개의 독립 프로세스를 같은 서버에서 기동한다 (architecture.md §2):

1. vLLM (8000): `serving/start_vllm.sh`
2. 임베딩 서버 (8001): `python serving/embedding_server.py`
3. 리랭커 서버 (8002): `python serving/reranker_server.py`
4. 본 서버 (9000): `uvicorn main:app --port 9000` — **단일 워커 강제** (Milvus Lite 파일 락)

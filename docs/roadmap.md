# roadmap.md — 단계별 구현 순서

각 단계는 완료 기준(DoD)을 만족해야 다음으로 넘어간다.
개발 노트북(RTX 4050)에서는 GGUF Q4_K_M + Ollama/llama.cpp로 프롬프트와
로직을 검증하고, tool-calling 관련 로직은 최종적으로 L40 + vLLM에서 재검증한다.

## 0단계 — 프로젝트 골격

작업:
- 디렉터리 구조 생성 (architecture.md §3 그대로)
- pyproject.toml (src 레이아웃), requirements.txt, Makefile, .env.example, langgraph.json
- shared/config.py: .env 로딩, frozen dataclass, get_config() 캐시 getter
- shared/llm_client.py: get_llm() 싱글턴
- ruff 설정, pytest 설정 (integration 마커 기본 skip)

완료 기준:
- `make lint`, `make test` 통과 (테스트 0개여도 수집 에러 없이)
- `python -c "from ax_rag.shared.config import get_config; print(get_config())"` 동작
- config에 localhost가 아닌 기본 URL이 하나도 없음

## 1단계 — 서빙 인프라 (serving/)

작업:
- embedding_server.py: BGE-M3 로드 (로컬 경로), `POST /embed`, `/health`
- reranker_server.py: bge-reranker-v2-m3 로드, `POST /rerank`, `/health`
- start_vllm.sh: interfaces.md §11 명령
- 두 FastAPI 서버 모두 로컬 경로에서만 모델 로드, timeout, 배치 처리

완료 기준:
- 헬스체크 통과, /embed가 1024차원 벡터 반환, /rerank가 0~1 점수 반환
- HF_HUB_OFFLINE=1 상태에서 기동 성공 (네트워크 시도 없음)

## 2단계 — 인덱싱 파이프라인

작업:
- shared/vectorstore.py, shared/parent_store.py (스키마는 interfaces.md §2,
  doc_classification 예약 필드 포함)
- indexer_graph/chunking.py: chunk_document, chunk_document_by_sections, chunk_parent_child
- shared/bm25_store.py: build/load/search
- indexer_graph/graph.py: chunk → embed_and_upsert
- scripts/bulk_ingest.py

완료 기준:
- 한국어 샘플 문서 2개(섹션 있는 것 1, 없는 것 1)를 적재하면
  company_docs, document_parents, bm25 인덱스 파일 모두 생성됨
- 유닛 테스트: 청킹 경계(섹션 혼입 방지), 부모-자식 parent_id 참조 무결성,
  맥락 헤더 부착, 한국어 separators 동작

## 3단계 — 검색 + 리랭크 축소 그래프

작업:
- retrieval_graph/acl.py: build_acl_filter_expr + filter_by_acl
- nodes/dense_retrieve.py, nodes/bm25_retrieve.py, fusion.py + nodes/fuse.py, nodes/rerank.py
- generate/verify 없이 route(더미 통과) → 검색 → 융합 → 리랭크만 연결한 축소 그래프

완료 기준:
- 샘플 질의에 대해 상위 5개가 육안으로 관련 있음
- 유닛 테스트: rrf_fuse 순위 결합 정확성, filter_by_acl가 타 부서
  DEPT_ONLY 청크를 제거함 (보안 테스트, 반드시 작성)
- bm25 인덱스 파일 삭제 후에도 dense 단독으로 정상 동작 (폴백 확인)

## 4단계 — 라우터 + 생성 + 검증

작업:
- prompts.py: 라우터/생성/검증 프롬프트. document delimiter와
  인젝션 방어 지시 포함 (interfaces.md §7)
- nodes/router.py: ClassifyAndRewrite tool-call, 실패 시 원본 + GENERAL 폴백
- retrieval_graph/budget.py: trim_history
- nodes/generate.py: 원본 질문 + rewritten_query 둘 다 프롬프트에
- nodes/verify.py: rule_based_verify 1차 → LLM VerifyAnswer 2차, fail-closed
- graph.py: 전체 조립, 조건부 엣지 (finalize / increment_retry / fallback)

완료 기준:
- 멀티턴 시나리오("육아휴직 알려줘" → "그거 얼마나 쓸 수 있어?")에서
  rewritten_query가 맥락을 해소함
- 근거 없는 답변을 강제로 만들었을 때 verify가 fail-closed로 fallback 도달
- 유닛 테스트: trim_history 상한 준수, rule_based_verify의 숫자/날짜 검출
- 주의: tool-calling은 개발 노트북(llama.cpp)과 L40(vLLM)의 파서 동작이
  다를 수 있음. L40 재검증 전까지 "노트북 통과"는 잠정 통과로 취급

## 5단계 — main.py 연동 (SSE)

작업:
- FastAPI `POST /query` — SSE StreamingResponse (interfaces.md §5 계약), `/health`
- to_internal_history: 미들웨어 role(human/ai) → 내부 role(user/assistant) 변환
- stream_answer: verify 통과 후 확정 답변을 문장 단위 text 이벤트로 분할 전송
  → sources 1회 → `{"type":"done"}` 종료 이벤트
- 예외 처리: 파이프라인 예외 시 error 이벤트 + done 이벤트로 스트림 정상 종료
- `X-Accel-Buffering: no` 헤더
- shared/audit_log.py 연결: 모든 질의 기록
- 단일 워커 강제 (실행 스크립트/문서에 명시, Milvus Lite 파일 락)

완료 기준:
- `curl -N`으로 이벤트 순서 확인: text 1개 이상 → sources 1회 → done 이벤트
- user_department 누락 요청 시 visibility ALL 문서만 검색됨 (제한적 폴백 테스트)
- 서비스 하나를 내려놓고 요청 시 error 이벤트 후 done 이벤트로 종료 (행 걸림 없음)
- audit_log.jsonl에 timestamp, user_department, question, domain, sources, grounded 기록됨
- 유닛 테스트: to_internal_history role 변환, sse_event 프레임 형식,
  stream_answer 분할 경계

## 6단계 — 문서 갱신 + 평가

작업:
- scripts/reindex_document.py: source_doc 기준 삭제 → 재적재 → BM25 전체 재빌드
- scripts/evaluate_rag.py: RAGAS (faithfulness, answer_relevancy,
  context_precision, context_recall), evaluator는 로컬 vLLM 엔드포인트
- eval_sets/hr_sample.jsonl 골격

완료 기준:
- 문서 1개를 수정 재적재해도 다른 문서 검색 결과가 오염되지 않음
- 평가 스크립트가 리랭커 ON/OFF, dense-only vs 하이브리드 비교 실행 가능
- RRF k 스윕 (20/40/60/100) 비교 실행 가능
- 평가셋에 "의미 질의"와 "정확한 용어 질의"가 섞여 있음

## 7단계 — L40 검증 (배포 전, 코드 범위 밖 포함)

- tool-calling 실전 성공률 측정 (라우터/검증), 필요시 프롬프트 보강 또는
  tool 없이 JSON 파싱 방식으로 대체
- `vllm bench serve`로 동시성 파라미터 확정
- 네트워크 차단 상태에서 전체 스택 기동 → 아웃바운드 시도 0건 실측
- chars_per_token=2.2 근사를 실제 문서로 보정

## 미확정 항목 (구현 중 만나면 사용자에게 확인)

- HWP 파서 선택 — 결정 전까지 PDF/DOCX만 지원, HWP는 섹션 없이 통짜 처리
  (PDF는 pdfplumber==0.11.10으로 지원 완료 — 통짜 처리, 스캔본 미지원.
   DOCX는 필요 시 python-docx 추가 예정)
- user_department를 미들웨어가 실제로 보내는지 (현재 요청 예시에 없음.
  누락 시 visibility ALL만 검색하는 제한적 폴백으로 구현)
- sources의 `page` 필드 — 청크 메타데이터에 페이지 정보가 없음.
  페이지 추적 요구가 확정되면 인덱싱 스키마에 page 필드 추가 필요
- Milvus Lite 데이터 파일 실제 경로/권한
- 예상 동시 사용자 수 (max_num_seqs, 병렬화 여부 결정 변수)

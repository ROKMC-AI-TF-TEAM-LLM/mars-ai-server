# architecture.md — 시스템 구조

## 1. 전체 위치

```
[프론트엔드] --요청--> [미들웨어 백엔드] --요청--> [A.X RAG 서버 ← 본 프로젝트]
  화면 / 입력            인증, 세션, 라우팅         main.py -> retrieval_graph
```

본 프로젝트의 범위는 가장 오른쪽 "A.X RAG 서버" 하나다. 프론트엔드와
미들웨어는 이미 존재하는 시스템이며, 우리는 미들웨어로부터 요청을 받는
지점(`POST /query`)만 인터페이스로 정의한다.

- 인증: 내부망 신뢰 기반. 별도 토큰 검증 없음 (미들웨어가 인증 완료 후 호출)
- 프로토콜: **SSE 스트리밍** (미들웨어 기존 계약). 요청은 JSON,
  응답은 text/event-stream — text 청크, sources 1회, error, `[DONE]` 종료 신호.
  상세는 interfaces.md §5

## 2. 프로세스 배치 (L40 한 대)

네 개의 독립 프로세스가 같은 물리 서버에서 동작한다. LangGraph 코드는
아래 세 GPU 서비스를 HTTP로 호출하는 클라이언트일 뿐이며, 모델을
프로세스 안에 로드하지 않는다.

- vLLM (A.X 4.0 Light): 포트 8000, `vllm serve` 독립 프로세스, 약 37GB
- 임베딩 서버 (BGE-M3): 포트 8001, FastAPI 독립 프로세스, 약 1~2GB
- 리랭커 서버 (bge-reranker-v2-m3): 포트 8002, FastAPI 독립 프로세스, 약 1.6GB
- LangGraph 앱 (main.py): 포트 9000, GPU 미사용, 단일 uvicorn 워커

역할 구분(라우터/생성/검증)은 별도 모델이 아니라 하나의 vLLM 서버를
시스템 프롬프트만 바꿔 공유한다. tool-calling은 vLLM의
`--enable-auto-tool-choice --tool-call-parser hermes` 옵션 기반.

## 3. 디렉터리 구조

```
ax-rag-multiagent/
├── langgraph.json              # 그래프 등록 매니페스트
├── pyproject.toml              # src 레이아웃 패키징 (requires-python==3.11.*)
├── requirements.txt            # 프로덕션 의존성, 전부 == 고정
├── requirements-eval.txt       # 평가 전용 (ragas, datasets), 별도 설치
├── Makefile                    # test / lint / format / dev
├── README.md
├── .env.example
├── main.py                     # 사내 인트라넷 연동용 FastAPI 래퍼
├── eval_sets/
│   └── hr_sample.jsonl         # 평가용 질문-정답 쌍
├── scripts/
│   ├── bulk_ingest.py          # 여러 문서를 순회하며 indexer_graph 호출
│   ├── reindex_document.py     # 문서 갱신: 기존 청크 삭제 후 재적재 + BM25 재빌드
│   └── evaluate_rag.py         # RAGAS 기반 성능 평가
├── src/ax_rag/
│   ├── shared/
│   │   ├── config.py           # 환경변수 로딩, frozen dataclass, get_config()
│   │   ├── llm_client.py       # get_llm() 싱글턴 (lru_cache)
│   │   ├── vectorstore.py      # Milvus Lite 자식 청크 컬렉션
│   │   ├── parent_store.py     # 부모 청크 컬렉션 (인덱스 없음)
│   │   ├── bm25_store.py       # Kiwi 토큰화 + bm25s 인덱스
│   │   └── audit_log.py        # 질의 감사 로그 (JSONL append)
│   ├── indexer_graph/
│   │   ├── state.py            # IndexState
│   │   ├── chunking.py         # 구조 인식 분할 + 부모-자식 청킹
│   │   └── graph.py            # chunk -> embed_and_upsert -> END
│   └── retrieval_graph/
│       ├── state.py            # RetrievalState
│       ├── prompts.py          # 라우터/생성/검증 시스템 프롬프트
│       ├── acl.py              # ACL 필터 (표현식 + BM25 후처리)
│       ├── fusion.py           # RRF 융합
│       ├── budget.py           # 컨텍스트 토큰 예산 계산 + 대화 이력 절삭
│       ├── graph.py            # StateGraph 조립
│       └── nodes/
│           ├── router.py       # ClassifyAndRewrite tool-call
│           ├── dense_retrieve.py
│           ├── bm25_retrieve.py
│           ├── fuse.py
│           ├── rerank.py       # top_n=5 확정 + 부모 치환
│           ├── generate.py
│           └── verify.py       # LLM 검증 + 규칙 기반 검증 이중화
├── serving/
│   ├── start_vllm.sh
│   ├── embedding_server.py
│   └── reranker_server.py
└── tests/
    ├── unit_tests/
    └── integration_tests/
```

## 4. retrieval_graph 흐름 (10노드 + 조건부 분기)

```
route ─(SMALLTALK)→ smalltalk ──────────────────────────────────────→ END
  └──→ dense_retrieve → bm25_retrieve → fuse → rerank → generate → verify
                                                              ↑          │
                                                    (실패, 재시도 여유)  │
                                                       increment_retry ←─┤
                                                                          │
                                              (성공) finalize / (소진) fallback
```

노드별 책임:

1. **route** — 질문 + 대화 이력 → `rewritten_query` + `domain`을 한 번의
   tool-call(ClassifyAndRewrite)로. 멀티턴 맥락 해소 + 구어체 정규화 + 도메인 분류.
   tool_call 없으면 원본 질문 + GENERAL 폴백.
   SMALLTALK(인사/잡담) 분류 시 검색·검증을 건너뛰고 **smalltalk** 노드가
   직접 짧게 응답한다 (문서 근거 주장 금지, sources 비움, grounded=False)
2. **dense_retrieve** — rewritten_query 임베딩 → Milvus Lite 검색.
   ACL은 Milvus 스칼라 필터로 적용. top_k=20
3. **bm25_retrieve** — Kiwi 토큰화 → bm25s 검색 → **ACL 후처리 필터 필수** → top_k=20.
   인덱스 없으면 빈 리스트 반환 (dense 단독 폴백)
4. **fuse** — RRF 융합 (k=60 시작, 평가로 조정) → 상위 20
5. **rerank** — 리랭커 서버 호출 → top_n=5 확정 → 그 5개만 부모 청크로 치환
6. **generate** — 근거 기반 답변 생성. 프롬프트에는 원본 질문과 rewritten_query를
   **둘 다** 포함시켜 검색-생성 미스매치를 모델이 감지할 여지를 남긴다
7. **verify** — 이중 검증:
   - 1차 규칙 기반: draft_answer의 수치/날짜/문서명이 retrieved_chunks에 실재하는지
   - 2차 LLM 기반: VerifyAnswer tool-call. tool_call 실패 시 grounded=False (fail-closed)
8. **finalize / increment_retry / fallback** — 성공 시 확정, 실패 시
   MAX_VERIFY_RETRY(=1)까지 generate 재실행, 소진 시 안전한 대체 답변

dense_retrieve와 bm25_retrieve는 독립적이라 병렬 가능하지만,
구현 단순성을 위해 순차부터 시작한다.

## 5. indexer_graph 흐름 (2노드)

1. **chunk** — text (+선택적 sections) → chunks
2. **embed_and_upsert** — 임베딩 서버 호출 → Milvus insert + Kiwi 토큰화 후 bm25s 인덱스 갱신/저장

## 6. 청킹 전략

- 단계 1: 구조 인식 분할 — 파서가 헤딩을 추출하면 섹션 단위 우선. 없으면 전체를 한 섹션으로
- 단계 2: 토큰 기준 재귀 분할 — RecursiveCharacterTextSplitter,
  separators 우선순위: `\n##`, `\n###` → `\n\n` → `\n` → `다.` `요.` → `.`
  length_function은 문자수/2.2 근사 (한국어, 실측 보정 예정)
- 단계 3: 맥락 헤더 — 각 청크 앞에 `[문서명 > 섹션명]` 부착
- 단계 4: 부모-자식 — 자식 150~200토큰(임베딩/검색 대상), 부모 800~1,200토큰(생성 컨텍스트).
  부모는 인덱스 없는 별도 Milvus 컬렉션(document_parents)에 저장

## 7. 컨텍스트 토큰 예산 (A.X 4.0 Light 16,384 상한)

- 시스템 프롬프트: 약 300
- 검색 컨텍스트 (top_n=5 x 부모 약 1,000): 약 5,000
- 대화 이력: **상한 1,500** (budget.py에서 최근 턴부터 역순으로 채우고 초과분 절삭)
- 질문: 약 500
- 답변 생성 여유: 약 2,000
- 소계 약 9,300 → `--max-model-len 12288`로 운영 (8192는 여유 부족)

부모 크기 x top_n이 지배 변수. 청킹 파라미터를 바꾸면 이 표를 재계산한다.

## 8. 스트리밍 (SSE, 확정)

미들웨어 계약이 SSE이므로 main.py는 처음부터 StreamingResponse로 구현한다.
다만 **토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송**이다:

- 이유: generate → verify → (실패 시) 재생성 구조에서, 미들웨어 이벤트
  타입에 "이전 텍스트 취소" 신호가 없다. 토큰을 실시간으로 흘리면
  재시도 시 사용자가 이미 본 답변을 되돌릴 수 없다
- 흐름: 그래프를 invoke로 완주 → finalize/fallback의 확정 답변을
  문장 단위로 쪼개 text 이벤트로 순차 전송 → sources 1회 → [DONE]
- 파이프라인 예외(서비스 다운 등)만 error 이벤트. fallback 답변은 정상 text
- `X-Accel-Buffering: no` 헤더 필수
- 향후 진짜 토큰 스트리밍으로 가려면 미들웨어에 "reset" 류의 이벤트
  타입 추가 협의가 선행되어야 함. 노드 안에서 llm.invoke를 직접 호출하는
  현 구조는 그 전환을 대비해 유지한다

## 9. 문서 갱신/삭제 (scripts/reindex_document.py)

1. company_docs에서 해당 source_doc의 자식 청크 삭제
2. document_parents에서 부모 청크 삭제
3. indexer_graph로 재적재
4. BM25 인덱스는 부분 삭제 불가 → **전체 재빌드** (야간 배치 전제로 설계)

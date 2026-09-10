# architecture.md — 시스템 구조

## 1. 전체 위치

```
[프론트엔드] --요청--> [미들웨어 백엔드] --요청--> [A.X RAG 서버 ← 본 프로젝트]
  화면 / 입력            인증, 세션, 라우팅         main.py -> query_graph
```

본 프로젝트의 범위는 가장 오른쪽 "A.X RAG 서버" 하나다. 프론트엔드와
미들웨어는 이미 존재하는 시스템이며, 우리는 미들웨어로부터 요청을 받는
지점(`POST /query`)만 인터페이스로 정의한다.

- 인증: 내부망 신뢰 기반. 별도 토큰 검증 없음 (미들웨어가 인증 완료 후 호출)
- 프로토콜: **SSE 스트리밍** (미들웨어 기존 계약). 요청은 JSON,
  응답은 text/event-stream — text 청크, sources 1회, error, `{"type":"done"}` 종료 이벤트.
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
mars-ai-server/
├── langgraph.json              # 그래프 등록 매니페스트
├── pyproject.toml              # src 레이아웃 패키징 (requires-python==3.11.*)
├── requirements.txt            # 프로덕션 의존성, 전부 == 고정
├── requirements-eval.txt       # 평가 전용 (ragas, datasets), 별도 설치
├── Makefile                    # test / lint / format / dev
├── README.md
├── .env.example                # L40 운영용 / .env.dev.example: 개발 노트북용
├── main.py                     # 미들웨어 연동 FastAPI 래퍼 (SSE, 문서 관리 API)
├── docs/                       # architecture / interfaces / roadmap / code_guide / deploy_l40
├── eval_sets/
│   └── hr_sample.jsonl         # 평가용 질문-정답 쌍
├── sample_docs/                # 개발용 한국어 샘플 문서 (훈령·법령 PDF 등)
├── scripts/
│   ├── dev_setup.ps1           # 개발 노트북 부트스트랩 (venv·모델·llama.cpp·Milvus)
│   ├── bulk_ingest.py          # 여러 문서를 순회하며 indexer_graph 호출
│   ├── reindex_document.py     # 문서 갱신: 기존 청크 삭제 후 재적재 + BM25 재빌드
│   └── evaluate_rag.py         # RAGAS 기반 성능 평가
├── src/ax_rag/
│   ├── shared/
│   │   ├── config.py           # 환경변수 로딩, frozen dataclass, get_config()
│   │   ├── llm_client.py       # get_llm() 싱글턴 (lru_cache)
│   │   ├── vectorstore.py      # Milvus 자식 청크 컬렉션 (+문서 인벤토리)
│   │   ├── parent_store.py     # 부모 청크 컬렉션 (인덱스 없음)
│   │   ├── bm25_store.py       # Kiwi 토큰화 + bm25s 인덱스
│   │   ├── audit_log.py        # 질의 감사 로그 (JSONL append)
│   │   ├── health.py           # 딥 헬스체크 (의존 서비스 4종 집계)
│   │   ├── ingest_jobs.py      # 적재 작업 인메모리 레지스트리 (202 백그라운드)
│   │   └── logging_setup.py    # 통일 포맷 로거 팩토리
│   ├── indexer_graph/
│   │   ├── state.py            # IndexState
│   │   ├── chunking.py         # 구조 인식 분할 + 부모-자식 청킹
│   │   ├── loaders.py          # 확장자별 로더 (.md/.txt/.pdf)
│   │   ├── ingest.py           # 적재/삭제 공용 로직 + 직렬화 잠금
│   │   └── graph.py            # chunk -> embed_and_upsert -> END
│   └── query_graph/
│       ├── state.py            # QueryState
│       ├── prompts.py          # 에이전트/생성/검증/잡담 시스템 프롬프트
│       ├── tools.py            # 도구 레지스트리 (노드·설명·매처·상태문구·지연실행)
│       ├── agent_tools.py      # ReAct 행동 레지스트리 (이름·phase·관측 포매팅)
│       ├── retrieval.py        # 검색 파이프라인 함수화 + 근거 병합(예산 절단)
│       ├── thought.py          # 판단 근거 위생 처리 (검증 밖 노출 텍스트)
│       ├── query_hygiene.py    # 검색어 위생 (파일 요청 표현 제거)
│       ├── tool_fallback.py    # 구조화 호출 3단 안전망 (예시 기반 재시도)
│       ├── acl.py              # ACL 필터 (표현식 + BM25 후처리)
│       ├── fusion.py           # RRF 융합
│       ├── budget.py           # 컨텍스트 토큰 예산 계산 + 대화 이력 절삭
│       ├── graph.py            # StateGraph 조립 (ReAct 루프 배선·합성)
│       └── nodes/
│           ├── agent.py        # AgentAction: 다음 행동 1개 + 판단 근거 (ReAct 판단부)
│           ├── act.py          # 행동 실행·관측 누적 / 반려 되먹임 / 지연 도구 (ReAct 실행부)
│           ├── smalltalk.py    # 잡담 응답 (direct_answer 경로)
│           ├── discharge_days.py  # 전역일 D-day 계산 (결정적 코드 도구, 예시)
│           ├── dense_retrieve.py  # ↓ 네 노드는 retrieval.run_search가 함수로 호출
│           ├── bm25_retrieve.py
│           ├── fuse.py
│           ├── rerank.py       # top_n=5 확정 + 부모 치환
│           ├── generate.py
│           └── verify.py       # LLM 근거 검증 (fail-closed)
├── serving/
│   ├── start_vllm.sh           # L40 운영 서빙
│   ├── start_llm_dev.ps1       # 개발 노트북 llama.cpp 서빙 (vLLM 대체)
│   ├── embedding_server.py
│   ├── reranker_server.py
│   └── milvus-dev/             # 개발용 Docker Milvus 설정
└── tests/
    ├── unit_tests/
    └── integration_tests/
```

## 4. query_graph 흐름 — ReAct 에이전트 루프

에이전트가 다음 행동 하나를 정하고(agent) 실행한 뒤(act) 관측을 보고 다시
판단하는 구조다. 이전 배선(route + 실행 큐, plan-then-execute)은 제거했으며
전환 설계와 근거는 `docs/react_migration_plan.md`에 있다.

```
START → agent ⇄ act              (상한: MAX_AGENT_STEPS=3 / MAX_SEARCH_CALLS=2)
          ├─(근거 있음)──→ generate → verify ─┬ 통과   → finalize → deferred → END
          │                    ↑              ├ 재검색 → retry_search → agent
          │                    └──────────────┤ 재시도 → increment_retry
          │                                   ├ 근거 0 → knowledge_answer → END
          │                                   └ 소진   → fallback → END
          ├─(도구 답변·파일 예약만)→ finalize → deferred → END
          └─(도구도 검색도 없음)──→ direct_answer → deferred → END
```

1. **agent** — 다음에 할 행동 **하나**와 그 판단 근거(thought)를 한 번의 구조화
   호출(`AgentAction`)로 정한다. 행동: `search_documents` | 도구 | `finish`.
   에이전트는 **답변을 쓰지 않는다** — 근거를 모으는 행동만 고르고, 답변 작성은
   generate가, 근거 검증은 verify가 그대로 맡는다 (실측으로 균형이 잡힌 두
   프롬프트를 루프 안으로 옮기지 않기 위한 분업).
   결정적 안전장치: ① 요청 tool 필드 강제 경로는 LLM 없이 직행 ② 결정적
   매처(전역일·파일 저장)가 잡은 도구는 LLM 없이 확정하며, 짧은 질문이면
   그대로 종결(LLM 0회) ③ 구조화 호출 실패 시 1라운드에 한해 **원본 질문으로
   검색 강제**(= 기존 DOC_SEARCH 폴백) ④ 상한 소진 시 LLM에 묻지 않고 종료
2. **act** — 행동을 실행하고 **압축된 관측**을 스크래치패드에 쌓는다.
   검색은 `retrieval.run_search`(dense→bm25→fuse→rerank)를 함수로 호출하며,
   **검색 범위(ACL·도메인)는 상태에서만 읽는다** — LLM 인자로 받지 않는다.
   누적 근거는 항상 `merge_chunks`로 `RERANK_TOP_N`개까지만 남긴다(§7 예산 불변식).
   지연 도구(파일 저장)는 실행하지 않고 **예약만** 한다
3. **retry_search** — verify 반려 사유를 관측으로 실어 에이전트에게 되돌린다
   (재검색 1회). 기존 구조가 같은 청크로 다시 쓰는 것뿐이었던 한계를 푼다
4. **deferred** — 검증 통과 후 예약된 도구(HWP_EXPORT 등)를 실행해 확정 답변
   뒤에 코드로 이어 붙인다. fallback·knowledge_answer는 이 경로를 타지 않는다
   (검증 못 받은 답변을 파일로 만들지 않는다 — fail-closed)
5. **direct_answer** — 도구도 검색도 쓰지 않은 질문(인사·잡담·자기소개).
   smalltalk 노드가 그대로 이 자리를 맡는다 (grounded=False, sources 비움)

### 4-1. 검색·생성·검증 (에이전트가 부르는 것들)

- **검색** — `retrieval.run_search()`가 dense → bm25 → fuse → rerank를 한 번에
  수행한다. 그래프 노드가 아니라 함수이며, act 노드가 호출한다
  - dense: 임베딩 → Milvus 검색. ACL(visibility/부서)은 Milvus 스칼라 필터.
    top_k=SEARCH_TOP_K(.env, 기본 20). 도메인 한정은 **요청이 명시한 경우
    (requested_domain)에만** 적용하며, LLM은 검색 범위를 정하지 못한다
  - bm25: Kiwi 토큰화 → bm25s 검색(3배 오버샘플) → **ACL 후처리 필터 필수**.
    인덱스 없으면 빈 리스트 (dense 단독 폴백)
  - fuse: RRF 융합(k=60) → 상위 RERANK_TOP_K(기본 20)
  - rerank: 리랭커 서버 → top_n=5 확정 → 그 5개만 부모 청크로 치환
  - 여러 번 검색하면 `merge_chunks`가 합집합을 리랭크 점수로 재정렬해
    **RERANK_TOP_N개로 절단**한다 (§7 예산 불변식)
- **generate** — 근거 기반 답변 생성. 프롬프트에 원본 질문과 검색 쿼리를
  **둘 다** 넣어 검색-생성 미스매치를 모델이 감지할 여지를 남긴다
- **verify** — LLM 근거 검증 (VerifyAnswer):
  - 전제 검사(코드): 답변이 비었거나 근거 0건이면 LLM 호출 없이 grounded=False
  - 판정 실패·예외 시에도 grounded=False (fail-closed)
  - **판정 범위는 "답변에 적힌 내용"뿐이다.** 질문의 일부를 못 답했거나
    "문서에서 확인되지 않는다"고 밝힌 것은 반려 사유가 아니다 (실측 오탐 사례는
    prompts.VERIFY_SYSTEM_PROMPT 주석 참조)
  - 반려 시 지목된 문장만 덜어내고 **재검증**하는 부분 수용을 한 번 시도한다.
    정제본도 반드시 검증을 통과해야 확정된다
- **finalize / increment_retry / fallback** — 성공 시 도구 답변과 문서 답변을
  실행 순서로 합성해 확정 (코드 조립만 — verify 뒤 LLM 가공 금지),
  실패 시 MAX_VERIFY_RETRY(=1)까지 generate만 재실행, 소진 시 문서 파트만
  안전한 대체 답변으로 바꿔 합성
- **knowledge_answer** — 검색이 **근거를 하나도 못 찾았고** 도메인을 한정하지
  않았을 때, 정형 사과 문구 대신 LLM 자체 지식으로 답한다
  (`KNOWLEDGE_FALLBACK_ENABLED`로 끌 수 있음). **재검색을 다 쓴 뒤에** 분기한다 —
  문서로 답할 기회를 남겨 둔 채 검증 없는 경로로 내려가지 않는다.
  ⚠️ **이 경로는 verify를 거치지 않는다.** SSE `notice` 이벤트
  (`code=ungrounded_knowledge`)로 문서 근거가 없음을 알리고, 감사 로그에
  `answer_mode="knowledge"`로 남겨 사후 추적이 가능하게 한다.
  근거가 **있는데** 검증에서 떨어진 경우는 이 경로로 보내지 않는다 —
  모델이 수치를 지어냈다는 신호이므로 검증 없이 확정하면 안 된다 (fail-closed)

dense와 bm25는 독립적이라 병렬 가능하지만, 구현 단순성을 위해 순차부터 시작한다.

## 5. indexer_graph 흐름 (2노드)

1. **chunk** — text (+선택적 sections) → chunks
2. **embed_and_upsert** — 임베딩 서버 호출 → Milvus insert + Kiwi 토큰화 후 bm25s 인덱스 갱신/저장

## 6. 청킹 전략 (`indexer_graph/chunking.py`)

프로덕션에서 실제로 도는 경로는 `indexer_graph/graph.py:chunk`가 호출하는
`chunk_parent_child` 하나뿐이다. 그 안에서 아래 4단계가 순서대로 적용된다.

- **단계 1 — 구조 인식 섹션 분해**: 파서(`loaders.load_document`)가 헤딩을
  추출하면(현재 `.md`의 `## ` 헤딩만, `parse_markdown_sections`) 섹션 단위로
  먼저 나눈다. 헤딩이 없으면(`.txt`, `.pdf`, 헤딩 없는 `.md`) 문서 전체를
  섹션 1개로 취급해 통짜 처리한다. 섹션을 먼저 자르는 이유는 하위 토큰
  분할이 서로 다른 주제를 한 청크에 섞는 것을 막기 위해서다
  (`test_섹션_간_텍스트가_혼입되지_않는다`로 고정).
- **단계 2 — 토큰 기준 재귀 분할**: `RecursiveCharacterTextSplitter`를 쓰되
  `length_function`에 문자열 길이 대신 `approx_tokens`(문자수/2.2 올림,
  `shared/tokens.py`)을 넣어 토큰 단위로 자른다. separators 우선순위는
  `\n##` → `\n###` → `\n\n` → `\n` → `다.` → `요.` → `.` → `""`(최후 강제
  분할) — 마크다운 헤딩 경계를 최우선으로, 그다음 문단·줄바꿈, 마지막에
  한국어 문장 종결 어미 순으로 시도한다. `keep_separator="end"`라 "~다."/
  "~요."로 끝나는 문장이 다음 청크로 안 넘어가고 앞 청크 끝에 남는다.
  overlap은 `chunk_size_tokens // 4`로 상한을 둔다 (splitter가
  `overlap >= chunk_size`를 거부하기 때문).
- **단계 3 — 맥락 헤더 부착**: 각 청크 앞에 `[문서명]` 또는 섹션이 있으면
  `[문서명 > 섹션명]`을 붙인다. 청크만 단독으로 읽어도 어느 문서·섹션
  소속인지 알 수 있게 하기 위함이다.
- **단계 4 — 부모-자식 이중 청킹**:
  - 자식 150~200토큰(기본 175, 임베딩/검색 대상), overlap 30토큰
    (`_CHILD_OVERLAP_TOKENS`) — 검색 경계에서 잘린 문장으로 인한 손실을
    줄이기 위해 겹친다.
  - 부모 800~1,200토큰(기본 1,000, 생성 컨텍스트), overlap 없음 — LLM
    프롬프트에 그대로 들어가므로 중복 텍스트로 예산을 낭비하지 않는다.
    인덱스 없는 별도 Milvus 컬렉션(`document_parents`)에 저장한다.
  - 자식은 각자 `parent_id`로 자신이 속한 부모를 참조하며, 항상 그 부모
    텍스트의 부분 문자열이다(`test_자식_본문은_자기_부모_텍스트에_포함된다`).
    검색은 자식으로 하고, 검색된 자식을 부모 텍스트로 치환해 생성
    컨텍스트를 만든다(§7 예산표의 "부모 약 1,000"이 이 값).

**Milvus 길이 상한과의 관계**: 자식 `text`는 VARCHAR(4000), 부모
`parent_text`는 VARCHAR(8000)(`shared/vectorstore.py`, `shared/parent_store.py`).
토큰 근사(2.2자/토큰)가 다소 보수적이라 기본 크기(부모 1,000토큰 ≈ 2,200자)로는
상한에 크게 못 미치지만, 헤딩이 드문 문서에서 재귀 분할이 강제 분할(`""`)까지
밀리는 경우를 대비한 여유다(`test_부모는_한글_여유_8000자를_넘지_않는다`).

**파일 형식별 섹션 추출**(`loaders.py`): `.md`만 `## ` 헤딩을 파싱하고,
`.txt`/`.pdf`는 섹션 없이 통짜로 넘어온다. HWP/DOCX는 미지원(roadmap.md).

⚠️ `chunk_document`/`chunk_document_by_sections`(비-부모자식 평면 청크, 기본
400/60토큰)는 코드에 남아 있지만 **현재 프로덕션 경로에서는 호출되지 않는다**
— `test_chunking.py`에서만 쓰인다. 부모-자식 도입 이전 설계의 잔재이니
새 적재 경로에 쓰기 전에 이 문서를 갱신할 것.

청킹 파라미터(부모/자식 크기)를 바꾸면 §7 컨텍스트 토큰 예산표(특히 generate
호출의 "검색 컨텍스트" 약 5,000)를 재계산해야 한다.

## 7. 컨텍스트 토큰 예산 (A.X 4.0 Light 16,384 상한)

예산은 **호출별 최댓값**으로 본다. LLM 호출은 각각 독립 컨텍스트라 누적되지 않는다.

**generate 호출 (지배 변수)**

- 시스템 프롬프트: 약 300
- 검색 컨텍스트 (top_n=5 x 부모 약 1,000): 약 5,000
- 대화 이력: **상한 1,500** (budget.py에서 최근 턴부터 역순으로 채우고 초과분 절삭)
- 질문: 약 500
- 답변 생성 여유: 약 2,000
- 소계 약 9,300 → `--max-model-len 12288`로 운영 (8192는 여유 부족)

**verify 호출**: 시스템 300 + 근거 5,000 + 답변 약 1,000 + 출력 512 ≈ 6,800

**agent 호출 (ReAct 루프)**: 시스템 약 500 + 이력 ≤1,500 + 질문 500 +
스크래치패드 **≤1,200** + 출력 512 ≈ 4,200

- 스크래치패드 상한 = `MAX_AGENT_STEPS(3) x 관측 400자(≈180토큰)` + thought·행동 JSON
- 관측은 요약본이다 (문서명 + 150자 발췌 x 3). 근거 전문은 스크래치패드가
  아니라 `retrieved_chunks`에 쌓인다

★ **예산 불변식**: 검색을 몇 번 하든 `retrieved_chunks`는 합집합을 리랭크 점수로
재정렬한 뒤 `RERANK_TOP_N`개로 절단한다 (`retrieval.merge_chunks`).
**이 절단 덕분에 다중 검색이 generate 컨텍스트를 늘리지 않으며**, 지배 변수는
여전히 generate의 9,300이라 `--max-model-len 12288`을 유지한다.
이 불변식은 유닛 테스트로 고정되어 있다 (`test_agent_act.py`).

부모 크기 x top_n이 지배 변수. 청킹 파라미터를 바꾸면 이 표를 재계산한다.

## 8. 스트리밍 (SSE, 확정)

미들웨어 계약이 SSE이므로 main.py는 처음부터 StreamingResponse로 구현한다.
다만 **토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송**이다:

- 이유: generate → verify → (실패 시) 재생성 구조에서, 미들웨어 이벤트
  타입에 "이전 텍스트 취소" 신호가 없다. 토큰을 실시간으로 흘리면
  재시도 시 사용자가 이미 본 답변을 되돌릴 수 없다
- 흐름: 그래프를 invoke로 완주 → finalize/fallback의 확정 답변을
  문장 단위로 쪼개 text 이벤트로 순차 전송 → sources 1회 → done 이벤트
- 파이프라인 예외(서비스 다운 등)만 error 이벤트. fallback 답변은 정상 text
- `X-Accel-Buffering: no` 헤더 필수
- 향후 진짜 토큰 스트리밍으로 가려면 미들웨어에 "reset" 류의 이벤트
  타입 추가 협의가 선행되어야 함. 노드 안에서 llm.invoke를 직접 호출하는
  현 구조는 그 전환을 대비해 유지한다

### 8-1. 추론 근거 스트리밍 (ReAct)

답변 본문은 검증 후에야 나가지만, **에이전트의 판단 근거는 실시간으로 흘린다.**
사용자는 "검색하는 중..."이 아니라 "일수는 확인했는데 신청 절차가 안 나와서
다시 찾는 중"을 본다.

- 별도 LLM 호출이 없다. `AgentAction` 스키마의 `thought` 필드로 행동 결정과
  **한 번에** 받는다 (thought를 첫 필드로 두어 모델이 근거를 먼저 쓰게 하는
  부수 효과도 있다 — 행동 선택 자체의 품질이 올라간다)
- 발행 시점은 `agent` 노드 완료 직후다. 이때 행동은 정해졌지만 아직 실행 전이라
  **"설명한 뒤 행동한다"는 순서가 그래프 구조상 보장된다**
- 전송은 기존 `status` 이벤트의 **선택 필드**(`thought`, `step`)다. 새 이벤트
  타입도 새 stage 값도 만들지 않으므로, 미들웨어가 모르는 필드를 무시하면
  현행과 완전히 같이 동작한다 (interfaces.md §5)
- ⚠️ **thought는 verify를 거치지 않고 사용자에게 보이는 유일한 LLM 텍스트다.**
  SMALLTALK·HWP_DRAFT와 같은 등급으로 다룬다:
  프롬프트가 "규정 내용·수치를 쓰지 말 것"을 지시하고, `query_graph/thought.py`가
  길이(100자)·개행·꺾쇠를 코드로 강제하며, `config.STREAM_THOUGHTS`로 즉시 끌 수 있다.
  최종 답변(`final_answer`)에는 절대 합성되지 않는다

## 9. 문서 갱신/삭제 (scripts/reindex_document.py)

1. company_docs에서 해당 source_doc의 자식 청크 삭제
2. document_parents에서 부모 청크 삭제
3. indexer_graph로 재적재
4. BM25 인덱스는 부분 삭제 불가 → **전체 재빌드** (야간 배치 전제로 설계)

## 10. 적재 원본 보관 주체 (미들웨어)

관리자 페이지에서 올린 문서 **원본의 영속 보관 주체는 미들웨어(자기 DB)**다.
MARS는 인덱싱 엔진이자 임시 처리 공간이며, 생성 파일(§8 EXPORT_DIR)의
fetch-and-store와 대칭 구조다:

- 미들웨어가 원본을 사용자·업로드 이력과 매핑해 영속 저장하고, MARS로는
  `POST /documents`로 바이트를 relay한다 (전송 방향은 항상 미들웨어→MARS
  인바운드 — 에어갭 아웃바운드 금지와 정합)
- MARS는 받은 원본을 청킹·임베딩·색인(Milvus·BM25)하고, 원본은 `UPLOAD_DIR`에
  **임시 스테이징**한다. 색인 결과는 영속, 스테이징 원본은 임시(TTL 정리는
  향후 도입 — roadmap 미확정 항목)
- 서버 재구축(Milvus 초기화) 시 재적재 원천: 운영은 미들웨어가 원본 재전송,
  개발은 `scripts/bulk_ingest.py`로 sample_docs 재적재. MARS 로컬 원본 소실에
  의존하지 않는다
- `GET /documents`(관리 목록)는 ACL을 적용하지 않아 DEPT_ONLY 문서명까지
  노출하므로, 미들웨어가 관리자 권한 확인 후에만 프록시한다 (interfaces.md §5)

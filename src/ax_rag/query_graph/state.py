"""query_graph 상태 정의 (interfaces.md §3)."""

from typing import TypedDict


class QueryState(TypedDict):
    """질의응답 그래프 상태.

    question/conversation_history/user_department는 호출자 입력,
    나머지는 노드가 채우는 파생 값이다.
    """

    question: str  # 원본 질문 (generate 프롬프트용)
    conversation_history: list[dict] | None  # [{"role": "user"|"assistant", "content": str}]
    rewritten_query: str | None  # route가 생성한 검색용 쿼리 (대표 쿼리, 리랭크 기준)
    # 실제로 검색에 쓰이는 쿼리 목록. 첫 항목이 대표 쿼리(rewritten_query)다.
    # 단일 쿼리는 하나의 의미 이웃만 긁어 다면적 질문에서 근거가 한쪽으로 쏠린다
    # (실측: "조건·기간·신청 방법"을 물었는데 근거 청크 1건) — 측면별로 나눠
    # 검색해 후보 폭을 넓힌다. 비어 있으면 rewritten_query 하나로 검색한다
    search_queries: list[str] | None
    user_department: str
    # 요청이 명시한 검색 도메인 한정 (main.py에서 정규화). 빈 값이면 전 도메인 검색.
    # 검색 필터에 쓰이는 유일한 도메인 값
    requested_domain: str | None
    # 처리 경로 대표값(계획의 첫 항목, 로그·하위 호환용). 요청의 tool 필드가
    # 선설정하면 강제(라우터 분류 무시), 없으면 route가 분류해 채운다
    intent: str | None
    # 실행된 경로 기록: act 노드가 실행 순서대로 덧붙인다 (검색하면 DOC_SEARCH,
    # 도구를 쓰면 그 intent). 순서 = 최종 답변 합성 순서 (graph._compose_final)
    intents: list[str] | None
    # 도구 실행 결과 누적: [{"intent": str, "answer": str}]. finalize/fallback이 합성
    tool_answers: list[dict] | None
    # 도구가 생성한 파일 목록: [{"name": str, "url": str, "tool": str}]
    # main.py가 SSE file 이벤트로 내보낸다 (미들웨어 fetch-and-store 신호)
    generated_files: list[dict] | None
    domain: str | None  # (예약) 과거 라우터 도메인 분류 자리 — 현재 미사용
    dense_candidates: list[dict] | None  # dense 검색 top_k개
    bm25_candidates: list[dict] | None  # bm25 검색 top_k개 (ACL 후처리 완료분)
    retrieved_candidates: list[dict] | None  # RRF 융합 후 상위 20
    # [{"text", "source_doc", "parent_id", "chunk_id", "domain", ...}, ...]
    retrieved_chunks: list[dict] | None  # 리랭크 + 부모 치환 후 top_n개 [{"text", "source_doc"}]
    draft_answer: str | None
    grounded: bool | None
    verify_reason: str | None
    retry_count: int
    # increment_retry가 실어 보내는 직전 반려 사유. generate가 재작성 지시로 쓴다.
    # verify_reason과 내용은 같지만 소유가 다르다 — 이 값이 있으면 "재생성 중"이라는
    # 뜻이고, 1차 생성에는 존재하지 않는다
    retry_hint: str | None
    final_answer: str | None
    # 답변이 만들어진 경로. grounded 불리언 하나로는 "검증 통과"와 "검증 실패"만
    # 구분되어, 근거 없이 LLM 지식으로 답한 경우를 사후에 추적할 수 없다.
    # - "grounded"  : 검증 통과 (finalize)
    # - "knowledge" : 근거 0건 → LLM 자체 지식으로 답변 (knowledge_answer, 검증 미거침)
    # - "fallback"  : 검증 실패 또는 도메인 한정 검색 실패 → 정형 안내 문구
    # 도구 단독 경로(SMALLTALK 등)는 채우지 않는다 (문서 답변 경로가 아니다).
    # 감사 로그와 SSE notice 이벤트가 이 값을 쓴다
    answer_mode: str | None

    # ── ReAct 에이전트 루프 ────────────────────────────────────────────────
    # 지금까지 한 행동과 그 관측(요약본): [{"action","thought","observation"}].
    # ★ 근거 전문은 여기 넣지 않는다 — retrieved_chunks에 쌓고 여기에는 발췌만
    # 남긴다. 스크래치패드가 부풀면 16K 예산이 깨진다
    agent_scratchpad: list[dict] | None
    # 직전 라운드의 판단 근거. SSE status 이벤트로 사용자에게 표시된다.
    # ⚠️ verify를 거치지 않는 텍스트다 — thought.sanitize_thought로 위생 처리한다
    agent_thought: str | None
    agent_steps: int  # 소비한 판단 라운드 수 (MAX_AGENT_STEPS 상한)
    search_calls: int  # 소비한 검색 횟수 (MAX_SEARCH_CALLS 상한)
    searched_queries: list[str] | None  # 중복 검색 차단용
    # agent가 정한 다음 행동과 인자. act 노드가 실행하고 분기가 이 값을 읽는다
    next_action: str | None
    next_action_args: dict | None
    # 결정적으로 확정된 단독 행동이라 실행 후 LLM에 다시 묻지 않는다는 표시
    force_finish: bool | None
    # 검증 통과 후 실행할 도구 목록 (파일 저장 등). 검증 실패 시 실행되지 않는다
    deferred_actions: list[str] | None
    # verify 반려 사유를 에이전트에게 되돌린 적이 있는지 (재검색 1회 제한)
    verify_feedback_used: bool | None
    # 에이전트가 이미 검색한 검색어를 또 내놓아 중복 차단에 걸렸다는 표시.
    # 새 검색어를 못 내놓는 상태라는 뜻이므로 반려 되먹임도 걸지 않는다
    search_ideas_exhausted: bool | None
    # 감사 로그용 도구 호출 기록: [{"step","action","thought","query"}]
    tool_calls_log: list[dict] | None

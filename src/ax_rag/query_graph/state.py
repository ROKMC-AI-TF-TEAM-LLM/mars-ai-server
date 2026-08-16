"""query_graph 상태 정의 (interfaces.md §3)."""

from typing import TypedDict


class QueryState(TypedDict):
    """질의응답 그래프 상태.

    question/conversation_history/user_department는 호출자 입력,
    나머지는 노드가 채우는 파생 값이다.
    """

    question: str
    conversation_history: list[dict] | None  # [{"role": "user"|"assistant", "content": str}]
    rewritten_query: str | None  # 검색용 재작성 쿼리 (대표 쿼리, 리랭크 기준)
    search_queries: list[str] | None  # 검색에 쓸 쿼리 목록. 첫 항목이 rewritten_query
    user_department: str
    requested_domain: str | None  # 요청이 한정한 도메인. 빈 값이면 전 도메인 검색
    # 요청이 지정한 프로젝트. 빈 값이면 전사 문서만, 값이 있으면 전사 + 그 프로젝트.
    # ⚠️ 멤버십 검증은 미들웨어 책임이다 (user_department와 동일 신뢰 모델)
    project_id: str | None
    # 프로젝트 지침 (사용자 작성). <instructions> delimiter로 감싸 데이터로 전달한다.
    # 답변 생성·잡담·지식 답변에 적용되고, verify에는 절대 넣지 않는다 —
    # 사용자가 검증 기준을 바꾸면 안전장치가 무력화된다
    project_instructions: str | None
    intent: str | None  # 처리 경로 대표값. 요청의 tool 필드가 선설정하면 강제
    intents: list[str] | None  # 실행된 경로 기록 = 최종 답변 합성 순서
    tool_answers: list[dict] | None  # [{"intent": str, "answer": str}]
    generated_files: list[dict] | None  # [{"name": str, "url": str, "tool": str}]
    domain: str | None  # (예약) 과거 라우터 도메인 분류 자리 — 현재 미사용
    dense_candidates: list[dict] | None
    bm25_candidates: list[dict] | None  # ACL 후처리 완료분
    retrieved_candidates: list[dict] | None  # RRF 융합 결과
    retrieved_chunks: list[dict] | None  # 리랭크 + 부모 치환 후 확정 근거
    draft_answer: str | None
    grounded: bool | None
    verify_reason: str | None
    retry_count: int
    retry_hint: str | None  # 직전 반려 사유. 있으면 "재생성 중"이라는 뜻이다
    final_answer: str | None
    # "grounded"(검증 통과) | "knowledge"(근거 0건 → LLM 지식, 검증 미거침) |
    # "fallback"(검증 실패·범위 밖). 도구 단독 경로는 채우지 않는다
    answer_mode: str | None

    # ── ReAct 에이전트 루프 ────────────────────────────────────────────────
    # ★ 근거 전문은 retrieved_chunks에 쌓고 여기에는 발췌만 남긴다 (16K 예산)
    agent_scratchpad: list[dict] | None  # [{"action", "thought", "observation"}]
    agent_thought: str | None  # ⚠️ verify를 거치지 않는다 — sanitize_thought 필수
    agent_steps: int  # 소비한 판단 라운드 수 (MAX_AGENT_STEPS 상한)
    search_calls: int  # 소비한 검색 횟수 (MAX_SEARCH_CALLS 상한)
    searched_queries: list[str] | None  # 중복 검색 차단용
    next_action: str | None
    next_action_args: dict | None
    force_finish: bool | None  # 결정적으로 확정된 행동 — 실행 후 LLM에 다시 묻지 않는다
    deferred_actions: list[str] | None  # 검증 통과 후 실행할 도구. 실패 시 미실행
    verify_feedback_used: bool | None  # 반려 사유 되먹임 여부 (재검색 1회 제한)
    search_ideas_exhausted: bool | None  # 새 검색어를 못 내놓는 상태 — 되먹임도 생략
    tool_calls_log: list[dict] | None  # 감사 로그용 [{"step", "action", "thought", "query"}]

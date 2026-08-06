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
    # 처리 계획: route가 확정한 경로 목록. 대부분 1개, 복합 질문이면 여러 개.
    # 순서 = 최종 답변 합성 순서 (graph._compose_final)
    intents: list[str] | None
    # 남은 실행 큐 (도구 먼저, DOC_SEARCH는 마지막). 도구 노드가 자신을 지우며 소비
    pending_intents: list[str] | None
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

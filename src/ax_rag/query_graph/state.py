"""query_graph 상태 정의 (interfaces.md §3)."""

from typing import TypedDict


class QueryState(TypedDict):
    """질의응답 그래프 상태.

    question/conversation_history/user_department는 호출자 입력,
    나머지는 노드가 채우는 파생 값이다.
    """

    question: str  # 원본 질문 (generate 프롬프트용)
    conversation_history: list[dict] | None  # [{"role": "user"|"assistant", "content": str}]
    rewritten_query: str | None  # route가 생성한 검색용 쿼리
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
    final_answer: str | None

"""retrieval_graph 상태 정의 (interfaces.md §3)."""

from typing import TypedDict


class RetrievalState(TypedDict):
    """질의응답 그래프 상태.

    question/conversation_history/user_department는 호출자 입력,
    나머지는 노드가 채우는 파생 값이다.
    """

    question: str  # 원본 질문 (generate 프롬프트용)
    conversation_history: list[dict] | None  # [{"role": "user"|"assistant", "content": str}]
    rewritten_query: str | None  # route가 생성한 검색용 쿼리
    user_department: str
    domain: str | None  # config.DOMAINS 중 하나
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

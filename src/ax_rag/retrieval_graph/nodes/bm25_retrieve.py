"""bm25_retrieve 노드: Kiwi 토큰화 → bm25s 검색 → ACL 후처리 필터(필수).

인덱스가 없으면 빈 리스트를 반환해 dense 단독 폴백이 되게 한다.
도메인 한정은 요청이 명시한 경우(requested_domain)에만 적용한다
(dense_retrieve와 동일 정책 — 라우터 분류는 검색 범위를 제한하지 않는다).
"""

from __future__ import annotations

from ax_rag.retrieval_graph.acl import filter_by_acl
from ax_rag.retrieval_graph.state import RetrievalState
from ax_rag.shared.bm25_store import bm25_search
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 검색 깊이 (architecture.md §4: bm25 top_k=20)
TOP_K = 20

# ACL 필터로 걸러질 분량을 감안해 더 깊이 검색한 뒤 필터 후 top_k로 자른다
_OVERSAMPLE_FACTOR = 3


def bm25_retrieve(state: RetrievalState) -> dict:
    """BM25 검색 → filter_by_acl 후처리(우회 금지) → top_k=20."""
    query = state.get("rewritten_query") or state["question"]
    scope = state.get("requested_domain") or "GENERAL"  # GENERAL=도메인 제한 없음

    raw_results = bm25_search(query, top_k=TOP_K * _OVERSAMPLE_FACTOR)
    if not raw_results:
        logger.info("bm25 결과 없음 (인덱스 부재 또는 무매칭) → dense 단독 폴백")
        return {"bm25_candidates": []}

    filtered = filter_by_acl(raw_results, scope, state.get("user_department", ""))
    candidates = filtered[:TOP_K]
    logger.info("bm25 검색: 원시 %d건 → ACL 후 %d건", len(raw_results), len(candidates))
    return {"bm25_candidates": candidates}

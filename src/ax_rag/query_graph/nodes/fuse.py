"""fuse 노드: dense/bm25 후보를 RRF로 융합해 상위 RERANK_TOP_K개를 확정한다.

검색 쿼리가 여럿이면 **쿼리별로 RRF를 돌린 뒤** 가중 쿼터로 병합한다.
전체를 한 번에 RRF하면 여러 쿼리에 공통으로 걸린 문서(교집합)가 상위를 독식해,
측면별 문서를 넓히려던 다중 쿼리의 목적과 반대로 간다.
"""

from __future__ import annotations

from ax_rag.query_graph.fusion import merge_query_results, rrf_fuse
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

_RRF_K = 60  # RRF 관례값 (평가로 조정)


def _query_count(candidates: list[dict]) -> int:
    """후보에 실린 query_index로 검색 쿼리 수를 센다 (없으면 1 = 구형/단일 경로)."""
    return 1 + max((item.get("query_index", 0) for item in candidates), default=0)


def fuse(state: QueryState) -> dict:
    """RRF 융합 (k=60 시작, 평가로 조정) → 상위 RERANK_TOP_K(리랭커 입력 후보 수)."""
    dense = state.get("dense_candidates") or []
    bm25 = state.get("bm25_candidates") or []
    top_k = get_config().RERANK_TOP_K
    query_count = max(_query_count(dense), _query_count(bm25))

    if query_count == 1:
        # 단일 쿼리: 기존 경로 그대로 (동작 불변)
        fused = rrf_fuse(dense, bm25, k=_RRF_K, top_n=top_k)
        logger.info("RRF 융합: %d건", len(fused))
        return {"retrieved_candidates": fused}

    per_query = [
        rrf_fuse(
            [item for item in dense if item.get("query_index", 0) == index],
            [item for item in bm25 if item.get("query_index", 0) == index],
            k=_RRF_K,
            top_n=top_k,
        )
        for index in range(query_count)
    ]
    fused = merge_query_results(per_query, top_n=top_k)
    logger.info(
        "RRF 융합: 쿼리 %d개(%s) → 병합 %d건",
        query_count,
        "/".join(str(len(items)) for items in per_query),
        len(fused),
    )
    return {"retrieved_candidates": fused}

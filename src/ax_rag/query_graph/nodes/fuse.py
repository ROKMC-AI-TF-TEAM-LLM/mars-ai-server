"""fuse 노드: dense/bm25 후보를 RRF로 융합해 상위 20개를 확정한다."""

from __future__ import annotations

from ax_rag.query_graph.fusion import rrf_fuse
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def fuse(state: QueryState) -> dict:
    """RRF 융합 (k=60 시작, 평가로 조정) → 상위 20."""
    fused = rrf_fuse(
        state.get("dense_candidates") or [],
        state.get("bm25_candidates") or [],
        k=60,
        top_n=20,
    )
    logger.info("RRF 융합: %d건", len(fused))
    return {"retrieved_candidates": fused}

"""rerank 노드: 리랭커 서버 호출 → top_n=5 확정 → 부모 청크 치환.

자식 청크(검색 정밀도)로 순위를 매기고, 확정된 top_n만 생성 컨텍스트용
부모 청크로 치환한다 (architecture.md §4·§6).
"""

from __future__ import annotations

import requests

from ax_rag.query_graph.state import QueryState
from ax_rag.shared import parent_store
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _score_candidates(query: str, passages: list[str]) -> list[float]:
    """리랭커 서버 호출 (localhost, timeout 필수). passages와 같은 순서의 0~1 점수."""
    config = get_config()
    response = requests.post(
        config.RERANKER_SERVER_URL,
        json={"query": query, "passages": passages},
        timeout=config.HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["scores"]


def rerank(state: QueryState) -> dict:
    """리랭크 top_n=5 확정 후 그 5개만 부모 청크로 치환한다."""
    config = get_config()
    candidates = state.get("retrieved_candidates") or []
    if not candidates:
        return {"retrieved_chunks": []}

    query = state.get("rewritten_query") or state["question"]
    scores = _score_candidates(query, [c["text"] for c in candidates])

    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)

    retrieved_chunks: list[dict] = []
    seen_parent_ids: set[str] = set()
    for candidate, score in ranked:
        if len(retrieved_chunks) >= config.RERANK_TOP_N:
            break
        parent_id = candidate.get("parent_id") or ""
        # 같은 부모의 자식이 여럿 뽑히면 부모 텍스트가 중복되므로 한 번만 치환한다
        if parent_id in seen_parent_ids:
            continue
        parent_text = parent_store.get_parent(parent_id) if parent_id else ""
        retrieved_chunks.append(
            {
                # 부모가 없으면 자식 텍스트로 폴백 (컨텍스트 공백 방지)
                "text": parent_text or candidate["text"],
                "source_doc": candidate["source_doc"],
                "rerank_score": float(score),
            }
        )
        if parent_id:
            seen_parent_ids.add(parent_id)

    logger.info("리랭크: 후보 %d건 → 확정 %d건", len(candidates), len(retrieved_chunks))
    return {"retrieved_chunks": retrieved_chunks}

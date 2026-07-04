"""dense_retrieve 노드: rewritten_query 임베딩 → Milvus Lite 벡터 검색.

ACL은 Milvus 스칼라 필터로 적용한다 (architecture.md §4).
"""

from __future__ import annotations

import requests

from ax_rag.retrieval_graph.acl import build_acl_filter_expr
from ax_rag.retrieval_graph.state import RetrievalState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger
from ax_rag.shared.vectorstore import get_client, get_collection

logger = get_logger(__name__)

# 검색 깊이 (architecture.md §4: dense top_k=20)
TOP_K = 20

# 후속 노드(융합/리랭크/부모 치환/감사 로그)가 쓰는 필드
_OUTPUT_FIELDS = [
    "text",
    "parent_id",
    "source_doc",
    "domain",
    "owning_department",
    "visibility",
]


def _embed_query(query: str) -> list[float]:
    """임베딩 서버 호출 (localhost, timeout 필수)."""
    config = get_config()
    response = requests.post(
        config.EMBEDDING_SERVER_URL,
        json={"texts": [query]},
        timeout=config.HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def dense_retrieve(state: RetrievalState) -> dict:
    """벡터 검색 top_k=20. ACL은 Milvus 필터 표현식으로 강제한다."""
    query = state.get("rewritten_query") or state["question"]
    expr = build_acl_filter_expr(state.get("domain") or "GENERAL", state.get("user_department", ""))

    embedding = _embed_query(query)
    hits_per_query = get_client().search(
        get_collection(),
        data=[embedding],
        filter=expr,
        limit=TOP_K,
        output_fields=_OUTPUT_FIELDS,
    )

    candidates: list[dict] = []
    for hit in hits_per_query[0]:
        entity = hit.get("entity", {})
        candidates.append(
            {
                # pymilvus는 PK를 "id"가 아니라 실제 필드명(chunk_id) 키로 반환한다
                "chunk_id": hit["chunk_id"],
                "dense_score": float(hit["distance"]),
                **{field: entity.get(field) for field in _OUTPUT_FIELDS},
            }
        )
    logger.info("dense 검색: %d건 (filter=%s)", len(candidates), expr)
    return {"dense_candidates": candidates}

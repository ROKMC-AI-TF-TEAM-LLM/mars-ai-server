"""dense_retrieve 노드: rewritten_query 임베딩 → Milvus Lite 벡터 검색.

ACL(visibility/부서)은 Milvus 스칼라 필터로 강제한다 (architecture.md §4).
도메인 한정은 요청이 명시한 경우에만 적용한다 — LLM 분류로 검색 범위를
좁히면 분류-적재 불일치로 정답 문서가 배제된다.
"""

from __future__ import annotations

import requests

from ax_rag.query_graph.acl import build_acl_filter_expr
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger
from ax_rag.shared.vectorstore import get_client, get_collection

logger = get_logger(__name__)

# 후속 노드(융합/리랭크/부모 치환/감사 로그)가 쓰는 필드
_OUTPUT_FIELDS = [
    "text",
    "parent_id",
    "source_doc",
    "domain",
    "owning_department",
    "visibility",
]


def _embed_queries(queries: list[str]) -> list[list[float]]:
    """임베딩 서버 호출 (localhost, timeout 필수).

    서버가 texts 배열을 받으므로 쿼리가 늘어도 HTTP 왕복은 1회다.
    """
    config = get_config()
    response = requests.post(
        config.EMBEDDING_SERVER_URL,
        json={"texts": queries},
        timeout=config.HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def _search(embedding: list[float], expr: str, query_index: int) -> list[dict]:
    """Milvus 벡터 검색 실행 → 후보 dict 목록.

    query_index는 fuse가 쿼리별로 묶어 융합할 때 쓴다.
    """
    hits_per_query = get_client().search(
        get_collection(),
        data=[embedding],
        filter=expr,
        limit=get_config().SEARCH_TOP_K,
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
                "query_index": query_index,
                **{field: entity.get(field) for field in _OUTPUT_FIELDS},
            }
        )
    return candidates


def search_queries_of(state: QueryState) -> list[str]:
    """검색에 쓸 쿼리 목록 (dense/bm25 공용). 없으면 대표 쿼리 하나로 폴백한다."""
    queries = [q for q in (state.get("search_queries") or []) if str(q).strip()]
    return queries or [state.get("rewritten_query") or state["question"]]


def dense_retrieve(state: QueryState) -> dict:
    """벡터 검색 (쿼리마다 top_k=SEARCH_TOP_K).

    보안 필터는 항상 적용하고, 도메인은 요청이 명시한 경우에만 한정한다.
    """
    queries = search_queries_of(state)
    # 요청이 도메인을 안 정했으면 GENERAL(=도메인 절 없음)로 전 도메인 검색
    scope = state.get("requested_domain") or "GENERAL"
    expr = build_acl_filter_expr(scope, state.get("user_department", ""))

    embeddings = _embed_queries(queries)  # 쿼리가 여럿이어도 HTTP 1회
    candidates: list[dict] = []
    for index, embedding in enumerate(embeddings):
        candidates.extend(_search(embedding, expr, index))
    logger.info("dense 검색: 쿼리 %d개 → %d건 (filter=%s)", len(queries), len(candidates), expr)
    return {"dense_candidates": candidates}

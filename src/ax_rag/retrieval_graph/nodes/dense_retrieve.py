"""dense_retrieve 노드: rewritten_query 임베딩 → Milvus Lite 벡터 검색.

ACL(visibility/부서)은 Milvus 스칼라 필터로 강제한다 (architecture.md §4).
도메인 한정은 요청이 명시한 경우(requested_domain)에만 적용한다 —
라우터의 LLM 분류는 검색 범위를 제한하지 않는다 (분류-적재 불일치로
정답 문서가 배제되는 사고 방지, 실측 사례 있음).
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


def _search(embedding: list[float], expr: str) -> list[dict]:
    """Milvus 벡터 검색 실행 → 후보 dict 목록."""
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
    return candidates


def dense_retrieve(state: RetrievalState) -> dict:
    """벡터 검색 top_k=20. 보안 필터 항상 적용, 도메인은 요청 명시 시에만 한정."""
    query = state.get("rewritten_query") or state["question"]
    # 요청이 도메인을 안 정했으면 GENERAL(=도메인 절 없음)로 전 도메인 검색
    scope = state.get("requested_domain") or "GENERAL"
    expr = build_acl_filter_expr(scope, state.get("user_department", ""))

    candidates = _search(_embed_query(query), expr)
    logger.info("dense 검색: %d건 (filter=%s)", len(candidates), expr)
    return {"dense_candidates": candidates}

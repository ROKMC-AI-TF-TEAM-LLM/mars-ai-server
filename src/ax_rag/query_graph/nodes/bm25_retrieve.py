"""bm25_retrieve 노드: Kiwi 토큰화 → bm25s 검색 → ACL 후처리 필터(필수).

인덱스가 없으면 빈 리스트를 반환해 dense 단독 폴백이 되게 한다.
도메인 한정은 요청이 명시한 경우(requested_domain)에만 적용한다
(dense_retrieve와 동일 정책 — 라우터 분류는 검색 범위를 제한하지 않는다).
"""

from __future__ import annotations

from ax_rag.query_graph.acl import filter_by_acl
from ax_rag.query_graph.nodes.dense_retrieve import search_queries_of
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.bm25_store import bm25_search, corpus_size
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# ACL 필터로 걸러질 분량을 감안해 더 깊이 검색한 뒤 필터 후 top_k로 자른다
_OVERSAMPLE_FACTOR = 3

# 프로젝트 한정 검색은 코퍼스 전체를 훑는다.
#
# BM25는 Milvus 필터가 못 미쳐 **후처리로만** 거를 수 있는데(acl.filter_by_acl),
# 프로젝트 문서는 코퍼스에서 극소수라 상위 60건 안에 하나도 안 들어오는 일이
# 생긴다. 그러면 후처리로는 살릴 방법이 없어 BM25 축이 통째로 죽는다.
#
# 깊이를 올리는 비용은 사실상 없다 (실측: 1,525건 코퍼스에서 top_k=60이 0.6ms,
# 전체가 1.2ms). 상한은 메모리 폭주만 막는 용도다 — 코퍼스가 이 값을 넘길
# 만큼 커지면 프로젝트별 BM25 인덱스로 바꿔야 한다 (docs/tools.md 참조).
_PROJECT_SEARCH_DEPTH_CAP = 20000


def _search_depth(top_k: int, project_id: str) -> int:
    """후처리 필터를 감안한 BM25 검색 깊이.

    프로젝트가 지정되면 코퍼스 전체를 훑는다 (_PROJECT_SEARCH_DEPTH_CAP 상한).
    프로젝트 문서가 소수라 얕게 뽑으면 후처리 전에 이미 잘려 나가기 때문이다.
    """
    if not project_id:
        return top_k * _OVERSAMPLE_FACTOR
    return min(max(corpus_size(), top_k * _OVERSAMPLE_FACTOR), _PROJECT_SEARCH_DEPTH_CAP)


def bm25_retrieve(state: QueryState) -> dict:
    """BM25 검색 → filter_by_acl 후처리(우회 금지) → 쿼리마다 top_k=SEARCH_TOP_K."""
    queries = search_queries_of(state)
    scope = state.get("requested_domain") or "GENERAL"  # GENERAL=도메인 제한 없음
    top_k = get_config().SEARCH_TOP_K
    department = state.get("user_department", "")
    project_id = state.get("project_id") or ""

    search_depth = _search_depth(top_k, project_id)
    candidates: list[dict] = []
    raw_total = 0
    for index, query in enumerate(queries):
        raw_results = bm25_search(query, top_k=search_depth)
        raw_total += len(raw_results)
        if not raw_results:
            continue
        # ACL 후처리는 쿼리마다 반드시 적용한다 (우회 경로를 만들지 않는다)
        filtered = filter_by_acl(raw_results, scope, department, project_id)
        candidates.extend({**item, "query_index": index} for item in filtered[:top_k])

    if not candidates:
        logger.info("bm25 결과 없음 (인덱스 부재 또는 무매칭) → dense 단독 폴백")
        return {"bm25_candidates": []}

    logger.info(
        "bm25 검색: 쿼리 %d개, 원시 %d건 → ACL 후 %d건", len(queries), raw_total, len(candidates)
    )
    return {"bm25_candidates": candidates}

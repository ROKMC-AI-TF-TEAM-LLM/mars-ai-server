"""검색 파이프라인 함수화: dense → bm25 → fuse → rerank를 한 번의 호출로.

에이전트가 "검색"을 도구 하나로 부르기 위한 조립일 뿐이며, 로직을 재작성하지
않는다 — 기존 노드 함수를 순서대로 호출한다.
"""

from __future__ import annotations

from ax_rag.query_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.query_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.query_graph.nodes.fuse import fuse
from ax_rag.query_graph.nodes.rerank import rerank
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def run_search(state: QueryState, query: str) -> list[dict]:
    """쿼리 하나로 검색 파이프라인을 완주해 근거 청크를 돌려준다.

    호출자의 상태를 바꾸지 않는다 (사본에서 진행) — 반복 검색 시 이전
    중간 산출물이 섞이지 않게 하기 위해서다.

    ⚠️ 검색 범위는 호출자(LLM)가 아니라 **상태가 정한다.** user_department나
    도메인을 인자로 받는 경로를 만들지 말 것 (CLAUDE.md 보안 규칙).
    """
    working: dict = {**state, "rewritten_query": query, "search_queries": [query]}
    working.update(dense_retrieve(working))
    working.update(bm25_retrieve(working))
    working.update(fuse(working))
    working.update(rerank(working))
    return working.get("retrieved_chunks") or []


def merge_chunks(existing: list[dict], new: list[dict], limit: int | None = None) -> list[dict]:
    """기존 근거와 새 근거를 합쳐 리랭크 점수 순 상위 limit개만 남긴다.

    ★ 예산 불변식: 검색을 몇 번 하든 generate가 보는 근거는 limit개를 넘지
    않는다. 이 절단이 없으면 재검색마다 컨텍스트가 불어나 16K 예산이 깨진다.

    중복 판정은 본문 텍스트로 한다 — 리랭크 산출물에 parent_id가 없기 때문이며,
    같은 부모가 다시 뽑히면 텍스트가 같으므로 결과는 부모 중복 제거와 같다.
    """
    top_n = limit if limit is not None else get_config().RERANK_TOP_N
    merged: list[dict] = []
    seen: set[str] = set()
    for chunk in [*existing, *new]:
        text = str(chunk.get("text") or "")
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(chunk)
    # 점수가 없는 항목(구형 상태·테스트 픽스처)은 뒤로 보낸다
    merged.sort(key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)
    if len(merged) > top_n:
        logger.debug("근거 병합: %d건 → 상위 %d건으로 절단", len(merged), top_n)
    return merged[:top_n]

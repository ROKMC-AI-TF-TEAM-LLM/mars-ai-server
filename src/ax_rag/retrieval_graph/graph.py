"""retrieval_graph 조립 (architecture.md §4).

3단계 축소 그래프: route(더미) → dense → bm25 → fuse → rerank → END.
4단계에서 route를 ClassifyAndRewrite tool-call로 교체하고
generate/verify/finalize·조건부 엣지를 추가한다.

dense와 bm25는 독립이라 병렬 가능하지만 구현 단순성을 위해 순차로 시작한다.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ax_rag.retrieval_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.retrieval_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.retrieval_graph.nodes.fuse import fuse
from ax_rag.retrieval_graph.nodes.rerank import rerank
from ax_rag.retrieval_graph.state import RetrievalState

logger = logging.getLogger(__name__)


def route(state: RetrievalState) -> dict:
    """더미 라우터 (3단계): 원본 질문 그대로 + GENERAL 폴백.

    4단계에서 ClassifyAndRewrite tool-call(멀티턴 맥락 해소 + 도메인 분류)로 교체한다.
    """
    return {
        "rewritten_query": state.get("rewritten_query") or state["question"],
        "domain": state.get("domain") or "GENERAL",
        "retry_count": state.get("retry_count") or 0,
    }


def _build_graph() -> StateGraph:
    builder = StateGraph(RetrievalState)
    builder.add_node("route", route)
    builder.add_node("dense_retrieve", dense_retrieve)
    builder.add_node("bm25_retrieve", bm25_retrieve)
    builder.add_node("fuse", fuse)
    builder.add_node("rerank", rerank)

    builder.add_edge(START, "route")
    builder.add_edge("route", "dense_retrieve")
    builder.add_edge("dense_retrieve", "bm25_retrieve")
    builder.add_edge("bm25_retrieve", "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_edge("rerank", END)
    return builder


graph = _build_graph().compile()

"""retrieval_graph 전체 조립 (architecture.md §4).

route → dense_retrieve → bm25_retrieve → fuse → rerank → generate → verify
verify 후 조건부 분기:
- grounded=True            → finalize (확정)
- 실패 + 재시도 여유 있음  → increment_retry → generate 재실행
- 실패 + 재시도 소진       → fallback (안전한 대체 답변)

dense와 bm25는 독립이라 병렬 가능하지만 구현 단순성을 위해 순차로 시작한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ax_rag.retrieval_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.retrieval_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.retrieval_graph.nodes.fuse import fuse
from ax_rag.retrieval_graph.nodes.generate import generate
from ax_rag.retrieval_graph.nodes.rerank import rerank
from ax_rag.retrieval_graph.nodes.router import route
from ax_rag.retrieval_graph.nodes.verify import verify
from ax_rag.retrieval_graph.prompts import FALLBACK_ANSWER
from ax_rag.retrieval_graph.state import RetrievalState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def finalize(state: RetrievalState) -> dict:
    """검증 통과한 초안을 확정 답변으로 승격한다."""
    return {"final_answer": state.get("draft_answer") or ""}


def increment_retry(state: RetrievalState) -> dict:
    """검증 실패 시 재시도 횟수를 올리고 generate로 되돌아간다."""
    retry_count = (state.get("retry_count") or 0) + 1
    logger.info(
        "검증 실패 → 재생성 시도 %d회차 (사유: %s)", retry_count, state.get("verify_reason")
    )
    return {"retry_count": retry_count}


def fallback(state: RetrievalState) -> dict:
    """재시도 소진 시 안전한 대체 답변을 확정한다 (fail-closed의 종착지)."""
    logger.warning("재시도 소진 → fallback 답변 (사유: %s)", state.get("verify_reason"))
    return {"final_answer": FALLBACK_ANSWER}


def after_verify(state: RetrievalState) -> str:
    """verify 결과에 따른 분기: finalize / increment_retry / fallback."""
    if state.get("grounded"):
        return "finalize"
    if (state.get("retry_count") or 0) < get_config().MAX_VERIFY_RETRY:
        return "increment_retry"
    return "fallback"


def _build_graph() -> StateGraph:
    builder = StateGraph(RetrievalState)
    builder.add_node("route", route)
    builder.add_node("dense_retrieve", dense_retrieve)
    builder.add_node("bm25_retrieve", bm25_retrieve)
    builder.add_node("fuse", fuse)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_node("verify", verify)
    builder.add_node("finalize", finalize)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "route")
    builder.add_edge("route", "dense_retrieve")
    builder.add_edge("dense_retrieve", "bm25_retrieve")
    builder.add_edge("bm25_retrieve", "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        after_verify,
        {
            "finalize": "finalize",
            "increment_retry": "increment_retry",
            "fallback": "fallback",
        },
    )
    builder.add_edge("increment_retry", "generate")
    builder.add_edge("finalize", END)
    builder.add_edge("fallback", END)
    return builder


graph = _build_graph().compile()

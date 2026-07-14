"""query_graph 전체 조립 (architecture.md §4) — plan-then-execute.

route가 계획(intents)을 확정하면 실행 큐(pending_intents)를 따라 진행한다:

route ─(계획이 단독 전용 도구뿐)→ TOOL_NODES[도구] → END        (예: SMALLTALK)
  └─→ [도구₁ → 도구₂ → ...] → dense_retrieve → bm25_retrieve → fuse → rerank
        (tool_answers 누적)        └(계획에 DOC_SEARCH 없으면 도구 후 바로 finalize)
                                → generate → verify
verify 후 조건부 분기:
- grounded=True            → finalize (도구 답변 + 문서 답변을 계획 순서로 합성)
- 실패 + 재시도 여유 있음  → increment_retry → generate 재실행 (도구는 재실행 안 함)
- 실패 + 재시도 소진       → fallback (도구 답변은 유지, 문서 파트만 대체 답변)

도구 추가는 tools.py의 TOOL_NODES 등록만으로 배선된다 (code_guide §12 패턴 B).
합성은 verify 뒤의 코드 조립만 허용한다 — LLM으로 다듬으면 검증이 닿지 않는
곳에서 수치가 변형될 수 있다 (fail-closed 원칙).
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from ax_rag.query_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.query_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.query_graph.nodes.fuse import fuse
from ax_rag.query_graph.nodes.generate import generate
from ax_rag.query_graph.nodes.rerank import rerank
from ax_rag.query_graph.nodes.router import route
from ax_rag.query_graph.nodes.verify import verify
from ax_rag.query_graph.prompts import FALLBACK_ANSWER
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import (
    DOC_SEARCH,
    TERMINAL_ONLY_TOOLS,
    TOOL_NODES,
    execution_queue,
)
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _plan_of(state: QueryState) -> list[str]:
    """상태에서 계획을 읽는다. 구형 단일 intent 상태도 허용한다 (방어적)."""
    plan = state.get("intents")
    if plan:
        return list(plan)
    intent = state.get("intent") or DOC_SEARCH
    return [intent if intent in (DOC_SEARCH, *TOOL_NODES) else DOC_SEARCH]


def _compose_final(state: QueryState, doc_part: str) -> str:
    """도구 답변과 문서 파트를 계획(intents) 순서로 조립한다 (코드 조립만, LLM 금지)."""
    tool_answers = {
        item.get("intent"): str(item.get("answer") or "")
        for item in (state.get("tool_answers") or [])
    }
    parts = [
        doc_part if name == DOC_SEARCH else tool_answers.get(name, "")
        for name in (state.get("intents") or [])
    ]
    composed = "\n\n".join(part for part in parts if part)
    # 계획이 없는 구형 상태(테스트 등)는 문서 파트를 그대로 확정한다
    return composed or doc_part


def finalize(state: QueryState) -> dict:
    """검증 통과한 초안(+도구 답변)을 계획 순서로 합성해 확정한다."""
    return {"final_answer": _compose_final(state, state.get("draft_answer") or "")}


def increment_retry(state: QueryState) -> dict:
    """검증 실패 시 재시도 횟수를 올리고 generate로 되돌아간다."""
    retry_count = (state.get("retry_count") or 0) + 1
    logger.info(
        "검증 실패 → 재생성 시도 %d회차 (사유: %s)", retry_count, state.get("verify_reason")
    )
    return {"retry_count": retry_count}


def fallback(state: QueryState) -> dict:
    """재시도 소진 시 안전한 대체 답변을 확정한다 (fail-closed의 종착지).

    도구 답변은 결정적 코드 산출물이라 검증 실패와 무관하므로 유지하고,
    문서 파트만 대체 답변으로 바꿔 합성한다.
    """
    logger.warning("재시도 소진 → fallback 답변 (사유: %s)", state.get("verify_reason"))
    return {"final_answer": _compose_final(state, FALLBACK_ANSWER)}


def _make_tool_step(intent_name: str, tool_node: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """도구 노드를 계획 실행 단계로 감싼다.

    도구 함수의 기존 계약({"final_answer", ...} 반환)은 그대로 두고, 답변을
    tool_answers에 누적하며 실행 큐에서 자신을 지운다. 도구별 특수 코드 없이
    레지스트리 등록만으로 복합 계획에 편입된다.
    """

    def tool_step(state: QueryState) -> dict:
        delta = tool_node(state) or {}
        answer = str(delta.get("final_answer") or "")
        return {
            "tool_answers": [
                *(state.get("tool_answers") or []),
                {"intent": intent_name, "answer": answer},
            ],
            "pending_intents": [
                name for name in (state.get("pending_intents") or []) if name != intent_name
            ],
        }

    return tool_step


def next_step(state: QueryState) -> str:
    """실행 큐의 다음 단계: 도구 노드 | dense_retrieve(DOC_SEARCH) | finalize(큐 소진)."""
    pending = state.get("pending_intents")
    if pending is None:  # 구형 상태: 계획에서 실행 큐를 재구성
        pending = execution_queue(_plan_of(state))
    if not pending:
        return "finalize"
    if pending[0] == DOC_SEARCH:
        return "dense_retrieve"
    return pending[0]


def after_route(state: QueryState) -> str:
    """route 결과 분기: 단독 전용 도구는 종착 노드로, 그 외는 실행 큐를 따른다."""
    plan = _plan_of(state)
    if len(plan) == 1 and plan[0] in TERMINAL_ONLY_TOOLS:
        return plan[0]
    return next_step(state)


def after_verify(state: QueryState) -> str:
    """verify 결과에 따른 분기: finalize / increment_retry / fallback."""
    if state.get("grounded"):
        return "finalize"
    if (state.get("retry_count") or 0) < get_config().MAX_VERIFY_RETRY:
        return "increment_retry"
    return "fallback"


def _build_graph() -> StateGraph:
    builder = StateGraph(QueryState)
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

    # 도구 레지스트리 자동 배선: 노드 이름 = intent 값.
    # 단독 전용 도구는 종착(→ END), 그 외는 실행 큐를 따라 다음 단계로 이어진다
    composable = [name for name in TOOL_NODES if name not in TERMINAL_ONLY_TOOLS]
    step_targets = {
        **{name: name for name in composable},
        "dense_retrieve": "dense_retrieve",
        "finalize": "finalize",
    }
    for intent_name, tool_node in TOOL_NODES.items():
        if intent_name in TERMINAL_ONLY_TOOLS:
            builder.add_node(intent_name, tool_node)
            builder.add_edge(intent_name, END)
        else:
            builder.add_node(intent_name, _make_tool_step(intent_name, tool_node))
            builder.add_conditional_edges(intent_name, next_step, step_targets)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        after_route,
        {**step_targets, **{name: name for name in TERMINAL_ONLY_TOOLS if name in TOOL_NODES}},
    )
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

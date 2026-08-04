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
from ax_rag.query_graph.prompts import FALLBACK_ANSWER, FALLBACK_DOMAIN_SCOPED_TEMPLATE
from ax_rag.query_graph.stages import (
    NODE_BM25_RETRIEVE,
    NODE_DENSE_RETRIEVE,
    NODE_FALLBACK,
    NODE_FINALIZE,
    NODE_FUSE,
    NODE_GENERATE,
    NODE_INCREMENT_RETRY,
    NODE_RERANK,
    NODE_ROUTE,
    NODE_VERIFY,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import (
    DOC_SEARCH,
    POST_SEARCH_TOOLS,
    TERMINAL_ONLY_TOOLS,
    TOOL_NODES,
    plan_of,
    resolve_pending,
)
from ax_rag.shared.config import DOMAIN_LABELS, get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


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
    """검증 통과한 초안(+도구 답변)을 계획 순서로 합성해 확정한다.

    실행 큐에서 DOC_SEARCH를 지워, 남은 후처리 도구(POST_SEARCH_TOOLS)가
    after_finalize 분기로 이어질 수 있게 한다.
    """
    pending = [n for n in (state.get("pending_intents") or []) if n != DOC_SEARCH]
    return {
        "final_answer": _compose_final(state, state.get("draft_answer") or ""),
        "pending_intents": pending,
    }


def increment_retry(state: QueryState) -> dict:
    """검증 실패 시 재시도 횟수를 올리고, **반려 사유를 실어** generate로 되돌아간다.

    사유를 넘기지 않으면 재생성이 1차와 똑같은 프롬프트로 돌아 같은 실수를
    반복한다 (실측: 777자 → 768자, 동일 사유로 연속 탈락 후 fallback).
    retry_hint는 generate가 읽어 재작성 지시로 쓴다.
    """
    reason = str(state.get("verify_reason") or "").strip()
    retry_count = (state.get("retry_count") or 0) + 1
    logger.info("검증 실패 → 재생성 시도 %d회차 (사유: %s)", retry_count, reason or "(없음)")
    return {"retry_count": retry_count, "retry_hint": reason}


def _fallback_answer(state: QueryState) -> str:
    """상황에 맞는 대체 답변 문구를 고른다.

    도메인을 한정한 검색에서 **근거를 하나도 못 찾았으면** 그 사실을 알린다 —
    범위 밖이라 못 찾은 것을 "그런 문서가 없다"로 오해하지 않게 하기 위해서다.

    근거는 있었는데 검증에서 떨어진 경우(지어낸 수치 등)는 범위 탓이 아니므로
    기본 문구를 쓴다. 안 그러면 엉뚱한 원인을 안내하게 된다.
    """
    requested_domain = state.get("requested_domain") or ""
    if requested_domain and not (state.get("retrieved_chunks") or []):
        label = DOMAIN_LABELS.get(requested_domain, requested_domain)
        return FALLBACK_DOMAIN_SCOPED_TEMPLATE.format(domain_label=label)
    return FALLBACK_ANSWER


def fallback(state: QueryState) -> dict:
    """재시도 소진 시 안전한 대체 답변을 확정한다 (fail-closed의 종착지).

    도구 답변은 결정적 코드 산출물이라 검증 실패와 무관하므로 유지하고,
    문서 파트만 대체 답변으로 바꿔 합성한다.
    """
    logger.warning(
        "재시도 소진 → fallback 답변 (사유: %s, 검색 범위=%s, 근거 %d건)",
        state.get("verify_reason"),
        state.get("requested_domain") or "전체",
        len(state.get("retrieved_chunks") or []),
    )
    return {"final_answer": _compose_final(state, _fallback_answer(state))}


def _make_post_tool_step(
    intent_name: str, tool_node: Callable[[dict], dict]
) -> Callable[[dict], dict]:
    """후처리 도구 노드 래퍼: 확정된 final_answer 뒤에 도구 답변을 이어 붙인다.

    검색 파이프라인 뒤(finalize 후)에 실행되므로 도구는 state.final_answer
    (방금 검증·합성된 답변)를 입력으로 쓸 수 있다. 단독 실행(계획이 후처리
    도구뿐)이면 final_answer가 아직 없어 도구 답변만 확정된다.
    합성은 코드 조립만 — verify 뒤 LLM 가공 금지 원칙 유지.
    """

    def post_tool_step(state: QueryState) -> dict:
        delta = tool_node(state) or {}
        answer = str(delta.get("final_answer") or "")
        base = state.get("final_answer") or _compose_final(state, "")
        return {
            "final_answer": f"{base}\n\n{answer}" if base and answer else (answer or base),
            "pending_intents": [
                name for name in (state.get("pending_intents") or []) if name != intent_name
            ],
            # 도구가 만든 파일 정보는 SSE file 이벤트 재료로 누적 전달한다
            "generated_files": [
                *(state.get("generated_files") or []),
                *(delta.get("generated_files") or []),
            ],
        }

    return post_tool_step


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
            # 도구가 만든 파일 정보는 SSE file 이벤트 재료로 누적 전달한다
            "generated_files": [
                *(state.get("generated_files") or []),
                *(delta.get("generated_files") or []),
            ],
        }

    return tool_step


def next_step(state: QueryState) -> str:
    """실행 큐의 다음 단계: 도구 노드 | dense_retrieve(DOC_SEARCH) | finalize(큐 소진)."""
    pending = resolve_pending(state)
    if not pending:
        return NODE_FINALIZE
    if pending[0] == DOC_SEARCH:
        return NODE_DENSE_RETRIEVE
    return pending[0]


def after_route(state: QueryState) -> str:
    """route 결과 분기: 단독 전용 도구는 종착 노드로, 그 외는 실행 큐를 따른다."""
    plan = plan_of(state)
    if len(plan) == 1 and plan[0] in TERMINAL_ONLY_TOOLS:
        return plan[0]
    return next_step(state)


def after_finalize(state: QueryState) -> str:
    """finalize·후처리 도구 완료 후 분기: 남은 후처리 도구 실행 또는 종료.

    fallback 경로는 이 분기를 타지 않는다 (검증 실패 답변은 후처리하지 않음).
    """
    pending = state.get("pending_intents") or []
    if pending and pending[0] in POST_SEARCH_TOOLS:
        return pending[0]
    return END


def after_verify(state: QueryState) -> str:
    """verify 결과에 따른 분기: finalize / increment_retry / fallback."""
    if state.get("grounded"):
        return NODE_FINALIZE
    if (state.get("retry_count") or 0) < get_config().MAX_VERIFY_RETRY:
        return NODE_INCREMENT_RETRY
    return NODE_FALLBACK


def _build_graph() -> StateGraph:
    builder = StateGraph(QueryState)
    builder.add_node(NODE_ROUTE, route)
    builder.add_node(NODE_DENSE_RETRIEVE, dense_retrieve)
    builder.add_node(NODE_BM25_RETRIEVE, bm25_retrieve)
    builder.add_node(NODE_FUSE, fuse)
    builder.add_node(NODE_RERANK, rerank)
    builder.add_node(NODE_GENERATE, generate)
    builder.add_node(NODE_VERIFY, verify)
    builder.add_node(NODE_FINALIZE, finalize)
    builder.add_node(NODE_INCREMENT_RETRY, increment_retry)
    builder.add_node(NODE_FALLBACK, fallback)

    # 도구 레지스트리 자동 배선: 노드 이름 = intent 값.
    # - 단독 전용(TERMINAL_ONLY): 종착 노드 (→ END)
    # - 후처리(POST_SEARCH): finalize 뒤에 실행, 확정 답변에 이어 붙임
    # - 그 외(전처리): 실행 큐를 따라 검색 파이프라인 앞에서 순차 실행
    pre_tools = [
        name
        for name in TOOL_NODES
        if name not in TERMINAL_ONLY_TOOLS and name not in POST_SEARCH_TOOLS
    ]
    post_tools = [name for name in TOOL_NODES if name in POST_SEARCH_TOOLS]
    step_targets = {
        **{name: name for name in pre_tools},
        **{name: name for name in post_tools},
        NODE_DENSE_RETRIEVE: NODE_DENSE_RETRIEVE,
        NODE_FINALIZE: NODE_FINALIZE,
    }
    post_targets = {**{name: name for name in post_tools}, END: END}
    for intent_name, tool_node in TOOL_NODES.items():
        if intent_name in TERMINAL_ONLY_TOOLS:
            builder.add_node(intent_name, tool_node)
            builder.add_edge(intent_name, END)
        elif intent_name in POST_SEARCH_TOOLS:
            builder.add_node(intent_name, _make_post_tool_step(intent_name, tool_node))
            builder.add_conditional_edges(intent_name, after_finalize, post_targets)
        else:
            builder.add_node(intent_name, _make_tool_step(intent_name, tool_node))
            builder.add_conditional_edges(intent_name, next_step, step_targets)

    builder.add_edge(START, NODE_ROUTE)
    builder.add_conditional_edges(
        NODE_ROUTE,
        after_route,
        {**step_targets, **{name: name for name in TERMINAL_ONLY_TOOLS if name in TOOL_NODES}},
    )
    builder.add_edge(NODE_DENSE_RETRIEVE, NODE_BM25_RETRIEVE)
    builder.add_edge(NODE_BM25_RETRIEVE, NODE_FUSE)
    builder.add_edge(NODE_FUSE, NODE_RERANK)
    builder.add_edge(NODE_RERANK, NODE_GENERATE)
    builder.add_edge(NODE_GENERATE, NODE_VERIFY)
    builder.add_conditional_edges(
        NODE_VERIFY,
        after_verify,
        {
            NODE_FINALIZE: NODE_FINALIZE,
            NODE_INCREMENT_RETRY: NODE_INCREMENT_RETRY,
            NODE_FALLBACK: NODE_FALLBACK,
        },
    )
    builder.add_edge(NODE_INCREMENT_RETRY, NODE_GENERATE)
    # finalize 후 남은 후처리 도구가 있으면 실행, 없으면 종료.
    # fallback은 후처리 없이 바로 종료 — 검증 실패 답변은 파일 등으로 후처리하지 않는다
    builder.add_conditional_edges(NODE_FINALIZE, after_finalize, post_targets)
    builder.add_edge(NODE_FALLBACK, END)
    return builder


graph = _build_graph().compile()

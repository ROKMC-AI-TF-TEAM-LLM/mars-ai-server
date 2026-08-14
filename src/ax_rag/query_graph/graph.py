"""query_graph 전체 조립 (architecture.md §4) — ReAct 에이전트 루프.

    START → agent ⇄ act              (상한: MAX_AGENT_STEPS / MAX_SEARCH_CALLS)
              ├─(근거 있음)──→ generate → verify ─┬ 통과   → finalize → deferred → END
              │                    ↑              ├ 재검색 → retry_search → agent
              │                    └──────────────┤ 재시도 → increment_retry
              │                                   ├ 근거 0 → knowledge_answer → END
              │                                   └ 소진   → fallback → END
              ├─(도구 답변·파일 예약만)→ finalize → deferred → END
              └─(도구도 검색도 없음)──→ direct_answer → deferred → END

에이전트는 **근거를 모으는 행동만** 고른다. 답변은 generate가 쓰고 verify가 검사한다.

도구 추가는 tools.TOOL_NODES + agent_tools.AGENT_TOOLS 등록만으로 배선된다
(code_guide §12 패턴 C). 합성은 verify 뒤의 코드 조립만 허용한다 — LLM으로
다듬으면 검증이 닿지 않는 곳에서 수치가 변형된다 (fail-closed).

전환 기록은 docs/react_migration_plan.md 참조.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ax_rag.query_graph.agent_tools import ACTION_FINISH
from ax_rag.query_graph.nodes.act import act, retry_search, run_deferred
from ax_rag.query_graph.nodes.agent import agent
from ax_rag.query_graph.nodes.generate import generate
from ax_rag.query_graph.nodes.knowledge_answer import generate_knowledge_answer
from ax_rag.query_graph.nodes.smalltalk import smalltalk
from ax_rag.query_graph.nodes.verify import verify
from ax_rag.query_graph.prompts import FALLBACK_ANSWER, FALLBACK_DOMAIN_SCOPED_TEMPLATE
from ax_rag.query_graph.stages import (
    NODE_ACT,
    NODE_AGENT,
    NODE_DEFERRED,
    NODE_DIRECT_ANSWER,
    NODE_FALLBACK,
    NODE_FINALIZE,
    NODE_GENERATE,
    NODE_INCREMENT_RETRY,
    NODE_KNOWLEDGE_ANSWER,
    NODE_RETRY_SEARCH,
    NODE_VERIFY,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import DOC_SEARCH
from ax_rag.shared.config import DOMAIN_LABELS, get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _compose_final(state: QueryState, doc_part: str) -> str:
    """도구 답변과 문서 파트를 실행 순서(intents)로 조립한다 (코드 조립만, LLM 금지)."""
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
    """검증 통과한 초안(+도구 답변)을 실행 순서로 합성해 확정한다."""
    return {
        "final_answer": _compose_final(state, state.get("draft_answer") or ""),
        "answer_mode": "grounded",
    }


def increment_retry(state: QueryState) -> dict:
    """재시도 횟수를 올리고 **반려 사유를 실어** generate로 되돌아간다.

    사유를 안 넘기면 재생성이 같은 실수를 그대로 반복한다.
    """
    reason = str(state.get("verify_reason") or "").strip()
    retry_count = (state.get("retry_count") or 0) + 1
    logger.info("검증 실패 → 재생성 시도 %d회차 (사유: %s)", retry_count, reason or "(없음)")
    return {"retry_count": retry_count, "retry_hint": reason}


def _fallback_answer(state: QueryState) -> str:
    """상황에 맞는 대체 답변 문구를 고른다.

    도메인 한정 검색에서 근거가 0건이면 범위 탓임을 알린다. 근거가 있었는데
    검증에서 떨어진 경우는 범위 탓이 아니므로 기본 문구를 쓴다.
    """
    requested_domain = state.get("requested_domain") or ""
    if requested_domain and not (state.get("retrieved_chunks") or []):
        label = DOMAIN_LABELS.get(requested_domain, requested_domain)
        return FALLBACK_DOMAIN_SCOPED_TEMPLATE.format(domain_label=label)
    return FALLBACK_ANSWER


def fallback(state: QueryState) -> dict:
    """재시도 소진 시 안전한 대체 답변을 확정한다 (fail-closed의 종착지).

    도구 답변은 결정적 산출물이라 유지하고 문서 파트만 대체한다.
    """
    logger.warning(
        "재시도 소진 → fallback 답변 (사유: %s, 검색 범위=%s, 근거 %d건)",
        state.get("verify_reason"),
        state.get("requested_domain") or "전체",
        len(state.get("retrieved_chunks") or []),
    )
    return {
        "final_answer": _compose_final(state, _fallback_answer(state)),
        "answer_mode": "fallback",
    }


def _can_answer_from_knowledge(state: QueryState) -> bool:
    """지식 기반 답변(검증 없는 경로)을 허용할 상황인지 판정한다.

    스위치 on + 도메인 미지정 + 근거 0건을 모두 만족해야 한다.
    ⚠️ 근거 0건 조건이 핵심이다 — 근거가 있는데 검증에서 떨어진 것은 모델이
    지어냈다는 신호라, 검증 없는 경로로 내보내면 그대로 확정된다 (fail-closed).
    """
    if not get_config().KNOWLEDGE_FALLBACK_ENABLED:
        return False
    if state.get("requested_domain"):
        return False
    return not (state.get("retrieved_chunks") or [])


def knowledge_answer(state: QueryState) -> dict:
    """검색 근거가 없을 때 LLM 자체 지식으로 답한다 (검증 미거침).

    답변에는 SSE notice 경고가 따라붙는다 (api/pipeline.py). grounded는
    False로 유지하므로 출처는 붙지 않는다. 생성에 실패하면 정형 안내로
    되돌리며, answer_mode가 fallback이 되어 경고도 붙지 않는다.
    """
    answer = generate_knowledge_answer(state)
    if not answer:
        return {
            "final_answer": _compose_final(state, _fallback_answer(state)),
            "answer_mode": "fallback",
        }
    logger.warning(
        "검색 근거 0건 → 지식 기반 답변으로 응답 (질문=%s, 검증 미거침)",
        state.get("question"),
    )
    return {"final_answer": _compose_final(state, answer), "answer_mode": "knowledge"}


# ── ReAct 루프 분기 (AGENT_MODE=true) ─────────────────────────────────────


def _finish_target(state: QueryState) -> str:
    """루프 종료 후 갈 곳을 고른다.

    검색을 시도했으면 generate로 보낸다 — 근거 0건이어도 이 경로를 거쳐야
    verify의 fail-closed 판정이 knowledge_answer/fallback으로 이어진다.
    """
    if (state.get("retrieved_chunks") or []) or (state.get("search_calls") or 0):
        return NODE_GENERATE
    if (state.get("tool_answers") or []) or (state.get("deferred_actions") or []):
        return NODE_FINALIZE
    return NODE_DIRECT_ANSWER


def after_agent(state: QueryState) -> str:
    """agent 판단 후 분기: 행동을 실행하거나(act) 루프를 끝낸다."""
    if str(state.get("next_action") or ACTION_FINISH) == ACTION_FINISH:
        return _finish_target(state)
    return NODE_ACT


def after_act(state: QueryState) -> str:
    """act 실행 후 분기: 다시 판단하거나(agent) 루프를 끝낸다.

    라운드 상한을 다 썼으면 되묻지 않고 끝낸다 (LLM 호출만 낭비된다).
    검색 상한은 보지 않는다 — 남은 라운드로 다른 도구를 고를 수 있다.
    """
    if state.get("force_finish"):
        return _finish_target(state)
    if (state.get("agent_steps") or 0) >= get_config().MAX_AGENT_STEPS:
        return _finish_target(state)
    return NODE_AGENT


def _can_retry_search(state: QueryState) -> bool:
    """검증 반려를 에이전트에게 되돌려 재검색할 수 있는 상황인지 판정한다.

    한 번만 허용한다. 상한을 다 썼거나 에이전트가 새 검색어를 못 내놓는
    상태(search_ideas_exhausted)면 되돌려도 같은 결과라 걸지 않는다.
    """
    config = get_config()
    if not config.AGENT_VERIFY_FEEDBACK:
        return False
    if state.get("verify_feedback_used") or state.get("search_ideas_exhausted"):
        return False
    return (state.get("search_calls") or 0) < config.MAX_SEARCH_CALLS and (
        state.get("agent_steps") or 0
    ) < config.MAX_AGENT_STEPS


def after_verify(state: QueryState) -> str:
    """verify 결과 분기: finalize / retry_search / knowledge_answer / increment_retry / fallback.

    재검색이 knowledge_answer보다 앞이다 — 문서로 답할 기회를 다 쓰기 전에
    검증 없는 경로로 내려가면 찾을 수 있었던 근거를 버리게 된다.
    """
    if state.get("grounded"):
        return NODE_FINALIZE
    if _can_retry_search(state):
        return NODE_RETRY_SEARCH
    if _can_answer_from_knowledge(state):
        return NODE_KNOWLEDGE_ANSWER
    if (state.get("retry_count") or 0) < get_config().MAX_VERIFY_RETRY:
        return NODE_INCREMENT_RETRY
    return NODE_FALLBACK


def direct_answer(state: QueryState) -> dict:
    """도구도 검색도 쓰지 않은 질문에 답한다 (인사·잡담·자기소개).

    smalltalk 노드를 그대로 쓴다 — 정체성 프롬프트와 grounded=False 계약이
    이미 그 노드에 있다.
    """
    return smalltalk(state)


def _build_graph() -> StateGraph:
    """ReAct 루프 배선."""
    builder = StateGraph(QueryState)
    builder.add_node(NODE_AGENT, agent)
    builder.add_node(NODE_ACT, act)
    builder.add_node(NODE_GENERATE, generate)
    builder.add_node(NODE_VERIFY, verify)
    builder.add_node(NODE_RETRY_SEARCH, retry_search)
    builder.add_node(NODE_FINALIZE, finalize)
    builder.add_node(NODE_INCREMENT_RETRY, increment_retry)
    builder.add_node(NODE_FALLBACK, fallback)
    builder.add_node(NODE_KNOWLEDGE_ANSWER, knowledge_answer)
    builder.add_node(NODE_DIRECT_ANSWER, direct_answer)
    builder.add_node(NODE_DEFERRED, run_deferred)

    loop_targets = {
        NODE_ACT: NODE_ACT,
        NODE_AGENT: NODE_AGENT,
        NODE_GENERATE: NODE_GENERATE,
        NODE_FINALIZE: NODE_FINALIZE,
        NODE_DIRECT_ANSWER: NODE_DIRECT_ANSWER,
    }
    builder.add_edge(START, NODE_AGENT)
    builder.add_conditional_edges(NODE_AGENT, after_agent, loop_targets)
    builder.add_conditional_edges(NODE_ACT, after_act, loop_targets)
    builder.add_edge(NODE_GENERATE, NODE_VERIFY)
    builder.add_conditional_edges(
        NODE_VERIFY,
        after_verify,
        {
            NODE_FINALIZE: NODE_FINALIZE,
            NODE_RETRY_SEARCH: NODE_RETRY_SEARCH,
            NODE_KNOWLEDGE_ANSWER: NODE_KNOWLEDGE_ANSWER,
            NODE_INCREMENT_RETRY: NODE_INCREMENT_RETRY,
            NODE_FALLBACK: NODE_FALLBACK,
        },
    )
    builder.add_edge(NODE_RETRY_SEARCH, NODE_AGENT)
    builder.add_edge(NODE_INCREMENT_RETRY, NODE_GENERATE)
    # 지연 도구(파일 저장 등)는 확정된 답변 뒤에만 실행된다.
    # fallback·knowledge_answer는 이 경로를 타지 않는다 — 검증을 통과하지
    # 못한 답변을 파일로 만들지 않는다 (fail-closed)
    builder.add_edge(NODE_FINALIZE, NODE_DEFERRED)
    builder.add_edge(NODE_DIRECT_ANSWER, NODE_DEFERRED)
    builder.add_edge(NODE_DEFERRED, END)
    builder.add_edge(NODE_FALLBACK, END)
    builder.add_edge(NODE_KNOWLEDGE_ANSWER, END)
    return builder


graph = _build_graph().compile()

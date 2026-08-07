"""query_graph 전체 조립 (architecture.md §4) — ReAct 에이전트 루프.

    START → agent ⇄ act              (상한: MAX_AGENT_STEPS / MAX_SEARCH_CALLS)
              ├─(근거 있음)──→ generate → verify ─┬ 통과   → finalize → deferred → END
              │                    ↑              ├ 재검색 → retry_search → agent
              │                    └──────────────┤ 재시도 → increment_retry
              │                                   ├ 근거 0 → knowledge_answer → END
              │                                   └ 소진   → fallback → END
              ├─(도구 답변·파일 예약만)→ finalize → deferred → END
              └─(도구도 검색도 없음)──→ direct_answer → deferred → END

에이전트는 **근거를 모으는 행동만** 고른다. 답변은 generate가 쓰고 verify가
검사한다 — 실측으로 균형이 잡힌 두 프롬프트를 루프 안으로 옮기지 않기 위한 분업이다.

도구 추가는 tools.TOOL_NODES + agent_tools.AGENT_TOOLS 등록만으로 배선된다
(code_guide §12 패턴 C). 합성은 verify 뒤의 코드 조립만 허용한다 — LLM으로
다듬으면 검증이 닿지 않는 곳에서 수치가 변형될 수 있다 (fail-closed 원칙).

이전 배선(route + 실행 큐, plan-then-execute)은 제거했다. 설계 근거와 전환
기록은 docs/react_migration_plan.md에 있다.
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
    """도구 답변과 문서 파트를 실행 순서(intents)로 조립한다 (코드 조립만, LLM 금지).

    intents는 act 노드가 실행한 경로를 순서대로 기록한 목록이다
    (검색을 하면 DOC_SEARCH, 도구를 쓰면 그 intent가 한 번씩 들어간다).
    """
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
    return {
        "final_answer": _compose_final(state, _fallback_answer(state)),
        "answer_mode": "fallback",
    }


def _can_answer_from_knowledge(state: QueryState) -> bool:
    """지식 기반 답변(검증 없는 경로)을 허용할 상황인지 판정한다.

    세 조건을 **모두** 만족할 때만 참이다:
    1) 설정 스위치가 켜져 있다 (운영에서 코드 수정 없이 끌 수 있게)
    2) 도메인을 한정하지 않았다 — 한정 검색의 실패는 "문서에 없음"이 아니라
       "이 범위에 없음"이라, 범위를 넓히라고 안내하는 쪽이 옳다
    3) 근거 청크가 0건이다 — ⚠️ 가장 중요한 조건. 근거가 있는데 검증에서
       떨어진 경우는 **모델이 수치를 지어냈다는 신호**이므로, 검증 없는 경로로
       내보내면 지어낸 내용을 그대로 확정하게 된다 (fail-closed 원칙 유지)
    """
    if not get_config().KNOWLEDGE_FALLBACK_ENABLED:
        return False
    if state.get("requested_domain"):
        return False
    return not (state.get("retrieved_chunks") or [])


def knowledge_answer(state: QueryState) -> dict:
    """검색 근거가 없을 때 LLM 자체 지식으로 답한다 (검증 미거침).

    답변에는 SSE notice 이벤트로 "문서 근거 없음" 경고가 따라붙는다
    (api/pipeline.py). grounded는 False로 유지하므로 출처는 붙지 않는다.

    도구 답변(D-day 계산 등)이 있으면 함께 합성한다. 이때 경고는 답변 전체를
    가리키게 되어 결정적 도구 산출물까지 포함하지만, 경고가 과한 쪽이 안전하다.

    생성에 실패하면 정형 안내로 되돌린다 — answer_mode도 fallback이 되어
    경고 이벤트가 붙지 않는다.
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

    - 근거가 있거나 검색을 시도했으면 → generate (검색 0건이면 빈 초안 →
      verify fail-closed → knowledge_answer/fallback으로 이어진다. 이 경로를
      유지해야 "근거 0건" 판정이 종전대로 작동한다)
    - 검색은 없었지만 도구 답변·예약이 있으면 → finalize (도구 산출물만 확정)
    - 아무것도 없으면 → direct_answer (인사·잡담·자기소개)
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

    라운드 상한을 다 썼으면 에이전트에게 되묻지 않고 끝낸다 — 물어봐야 답은
    정해져 있고 LLM 호출만 한 번 더 나간다. 검색 상한은 여기서 보지 않는다:
    검색을 다 썼어도 남은 라운드로 다른 도구를 고를 수 있고, 검색 요청이
    들어오면 agent 노드가 종료로 바꾼다.
    """
    if state.get("force_finish"):
        return _finish_target(state)
    if (state.get("agent_steps") or 0) >= get_config().MAX_AGENT_STEPS:
        return _finish_target(state)
    return NODE_AGENT


def _can_retry_search(state: QueryState) -> bool:
    """검증 반려를 에이전트에게 되돌려 재검색할 수 있는 상황인지 판정한다.

    한 번만 허용한다 (verify_feedback_used). 상한을 이미 썼으면 되돌려도
    에이전트가 검색을 못 하므로 의미가 없다.

    에이전트가 방금 같은 검색어를 반복해 중복 차단에 걸렸다면
    (search_ideas_exhausted) 되돌려도 같은 검색어가 또 나온다 — 이 경우도
    되먹임을 걸지 않는다 (실측: 3라운드 내내 동일 검색어 반복).
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

    재검색을 **knowledge_answer보다 앞에** 둔다. 문서로 답할 기회를 다 쓰기
    전에 검증 없는 지식 답변으로 내려가면, 재검색으로 찾을 수 있었던 근거를
    버리게 된다. 그 외 순서와 fail-closed 원칙은 기존 분기와 같다.
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
    이미 그 노드에 있다. ReAct에서는 "잡담 도구를 실행"하는 대신 에이전트가
    아무 도구도 쓰지 않고 끝낸 경우가 이 경로다.
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

"""act 노드와 그 짝들: 에이전트가 고른 행동을 실제로 실행한다 (ReAct 루프의 실행부).

- act           — 검색 또는 도구를 실행하고 **압축된 관측**을 스크래치패드에 쌓는다
- retry_search  — verify 반려 사유를 관측으로 실어 에이전트에게 되돌린다
- run_deferred  — 검증 통과 후 지연 도구(파일 저장 등)를 실행한다

관측은 요약본이고, 답변 작성에 쓰이는 근거 전문은 state.retrieved_chunks에
쌓인다. 그리고 **누적 근거는 항상 RERANK_TOP_N개로 절단**된다
(retrieval.merge_chunks) — 이 절단이 다중 검색에도 생성 컨텍스트가 늘지 않는
이유이자 16K 예산이 유지되는 근거다.

보안: 검색 범위(ACL·도메인)는 상태에서만 읽는다. 에이전트가 준 인자는
검색어와 초안 본문뿐이며, 검색어조차 파일 요청 표현을 걷어낸 뒤 쓴다.
"""

from __future__ import annotations

import re

from ax_rag.query_graph.agent_tools import (
    ACTION_FINISH,
    ACTION_SEARCH,
    AGENT_TOOLS,
    DOC_SEARCH,
    PHASE_DEFERRED,
    format_deferred_observation,
    format_duplicate_observation,
    format_search_observation,
    format_tool_observation,
    format_verify_observation,
)
from ax_rag.query_graph.retrieval import merge_chunks, run_search
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    """중복 검색 판정용 정규화 (공백·대소문자 무시)."""
    return _WHITESPACE_RE.sub("", query).lower()


def _record(state: QueryState, action: str, observation: str, thought: str = "") -> list[dict]:
    """스크래치패드에 관측 한 건을 덧붙인 새 목록을 만든다."""
    # 다음 라운드의 판단 재료가 이것이다 — 관측이 부실하면 에이전트가 같은
    # 실수를 반복하므로, 무엇을 돌려줬는지 볼 수 있어야 한다
    logger.debug("관측[%s] ↓\n%s", action, observation)
    return [
        *(state.get("agent_scratchpad") or []),
        {"action": action, "thought": thought, "observation": observation},
    ]


def _log_entry(state: QueryState, action: str, args: dict) -> list[dict]:
    """감사 로그용 도구 호출 기록 (무엇을 어떤 인자로 불렀는가)."""
    entry = {
        "step": state.get("agent_steps") or 0,
        "action": action,
        "thought": state.get("agent_thought") or "",
    }
    if args.get("query"):
        entry["query"] = args["query"]
    return [*(state.get("tool_calls_log") or []), entry]


def _executed_paths(state: QueryState, intent: str) -> list[str]:
    """실행된 경로 기록(intents)에 intent를 한 번만 덧붙인다.

    최종 답변 합성 순서(graph._compose_final)가 이 목록을 따른다 —
    ReAct에서는 계획이 아니라 **실행 순서**가 합성 순서가 된다.
    """
    paths = list(state.get("intents") or [])
    if intent not in paths:
        paths.append(intent)
    return paths


def _act_search(state: QueryState, args: dict) -> dict:
    """검색 실행 → 근거 병합(절단) → 관측 누적."""
    config = get_config()
    query = str(args.get("query") or "").strip() or state["question"]
    searched = list(state.get("searched_queries") or [])
    normalized = _normalize_query(query)

    if normalized in {_normalize_query(item) for item in searched}:
        # 같은 검색을 반복하면 결과도 같다 — LLM에 되묻지 않고 루프를 끝낸다.
        # search_ideas_exhausted를 함께 세운다: 에이전트가 새 검색어를 못 내놓는
        # 상태라는 뜻이므로, 검증 반려 후 되먹임(retry_search)으로 루프를 다시
        # 열어도 같은 검색어가 또 나올 뿐이다 (실측: 3라운드 내내 동일 검색어)
        logger.info("중복 검색어 감지(%s) → 재검색 생략하고 종료", query)
        return {
            "agent_scratchpad": _record(state, ACTION_SEARCH, format_duplicate_observation(query)),
            "force_finish": True,
            "search_ideas_exhausted": True,
        }

    chunks = run_search(state, query)
    merged = merge_chunks(state.get("retrieved_chunks") or [], chunks, config.RERANK_TOP_N)
    logger.info(
        "검색 실행: query=%s, 신규 %d건 → 누적 %d건 (%d회차)",
        query,
        len(chunks),
        len(merged),
        (state.get("search_calls") or 0) + 1,
    )
    delta: dict = {
        "retrieved_chunks": merged,
        "search_calls": (state.get("search_calls") or 0) + 1,
        "searched_queries": [*searched, query],
        "intents": _executed_paths(state, DOC_SEARCH),
        "agent_scratchpad": _record(
            state,
            ACTION_SEARCH,
            format_search_observation(query, chunks, len(merged)),
            str(state.get("agent_thought") or ""),
        ),
        "tool_calls_log": _log_entry(state, ACTION_SEARCH, {"query": query}),
    }
    if not state.get("rewritten_query"):
        # 첫 검색어를 대표 쿼리로 남긴다 (리랭크 기준·로그·평가 스크립트가 참조)
        delta["rewritten_query"] = query
    return delta


def _act_tool(state: QueryState, action: str, args: dict) -> dict:
    """도구 실행. 지연 도구는 실행하지 않고 예약만 한다 (검증 통과 후 실행)."""
    tool = AGENT_TOOLS[action]

    if tool.phase == PHASE_DEFERRED:
        deferred = list(state.get("deferred_actions") or [])
        if action not in deferred:
            deferred.append(action)
        return {
            "deferred_actions": deferred,
            "intents": _executed_paths(state, tool.intent),
            "agent_scratchpad": _record(
                state,
                action,
                format_deferred_observation(action),
                str(state.get("agent_thought") or ""),
            ),
            "tool_calls_log": _log_entry(state, action, args),
        }

    delta = tool.node(state) or {}
    answer = str(delta.get("final_answer") or "")
    # 도구 계약(tool_contract.tool_answer)의 grounded=False·retrieved_chunks=[]는
    # **일부러 버린다.** 그 값들은 "도구 단독 응답"을 전제로 한 것이라, 루프에서
    # 그대로 상태에 병합하면 이미 모아 둔 검색 근거를 지워 버린다. 도구가 근거를
    # 주장하지 않는다는 성질은 grounded를 세우지 않는 것으로 이미 지켜진다
    # (verify를 통과해야만 True가 되고, 그때만 출처가 붙는다)
    return {
        "tool_answers": [
            *(state.get("tool_answers") or []),
            {"intent": tool.intent, "answer": answer},
        ],
        "intents": _executed_paths(state, tool.intent),
        "generated_files": [
            *(state.get("generated_files") or []),
            *(delta.get("generated_files") or []),
        ],
        "agent_scratchpad": _record(
            state,
            action,
            format_tool_observation(action, answer),
            str(state.get("agent_thought") or ""),
        ),
        "tool_calls_log": _log_entry(state, action, args),
    }


def act(state: QueryState) -> dict:
    """에이전트가 고른 행동을 실행하고 관측을 남긴다."""
    action = str(state.get("next_action") or ACTION_FINISH)
    args = dict(state.get("next_action_args") or {})

    if action == ACTION_SEARCH:
        return _act_search(state, args)
    if action in AGENT_TOOLS:
        return _act_tool(state, action, args)

    # 여기 오면 분기 설정이 어긋난 것이다 (agent가 finish를 냈으면 act로 오지 않는다)
    logger.warning("실행할 수 없는 행동: %r → 루프를 종료한다", action)
    return {"force_finish": True}


def retry_search(state: QueryState) -> dict:
    """verify 반려 사유를 관측으로 실어 에이전트에게 되돌린다 (재검색 1회).

    반려된 초안과 재시도 힌트를 함께 비운다 — 새 근거로 처음부터 다시 쓰게
    하기 위해서다. verify_feedback_used를 세워 이 경로가 한 번만 돌게 한다.
    """
    reason = str(state.get("verify_reason") or "").strip() or "사유 없음"
    logger.info("검증 반려 → 에이전트 재검색 시도 (사유: %s)", reason)
    return {
        "agent_scratchpad": _record(state, "verify", format_verify_observation(reason)),
        "verify_feedback_used": True,
        "draft_answer": "",
        "agent_thought": "",
    }


def run_deferred(state: QueryState) -> dict:
    """검증 통과 후 지연 도구를 순서대로 실행해 확정 답변 뒤에 이어 붙인다.

    합성은 코드 조립만 한다 — verify 뒤 LLM 가공 금지 원칙 유지.
    검증 실패(fallback·knowledge) 경로는 이 노드로 오지 않는다.
    """
    actions = list(state.get("deferred_actions") or [])
    if not actions:
        return {}

    answer = str(state.get("final_answer") or "")
    files = list(state.get("generated_files") or [])
    working: dict = dict(state)
    for action in actions:
        tool = AGENT_TOOLS.get(action)
        if tool is None:
            continue
        working["final_answer"] = answer
        delta = tool.node(working) or {}
        tool_answer = str(delta.get("final_answer") or "")
        files.extend(delta.get("generated_files") or [])
        answer = f"{answer}\n\n{tool_answer}" if answer and tool_answer else (tool_answer or answer)
        logger.info("지연 도구 실행: %s", action)

    return {"final_answer": answer, "generated_files": files, "deferred_actions": []}

"""그래프 노드 이름과 진행 상태(SSE status) 안내 문구.

노드 이름을 API 계층이 문자열로 알고 있으면, 노드 이름을 바꿨을 때
진행 안내가 조용히 사라진다 (테스트도 못 잡는다). 그래서 노드 이름 상수와
"이 노드가 끝나면 다음은 무엇인가"의 판단을 그래프 쪽에 둔다 —
graph.py가 노드를 등록할 때 쓰는 이름과 같은 상수를 여기서 쓴다.

status 이벤트의 stage 값("retrieve", "rerank", ...)은 미들웨어·프론트와의
계약이다 (interfaces.md §5, POST /query 응답 문서). 노드 이름과 별개이며
바꾸면 프론트 표시가 깨진다. **ReAct 루프에서도 stage 허용값은 늘어나지
않는다** — 같은 값이 여러 번 나갈 뿐이며, 계약은 이미 "status 0회 이상"이다.
"""

from __future__ import annotations

from ax_rag.query_graph.agent_tools import (
    ACTION_FINISH,
    ACTION_SEARCH,
    AGENT_TOOLS,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import (
    DEFAULT_TOOL_STATUS_MESSAGE,
    DOC_SEARCH,
    TERMINAL_ONLY_TOOLS,
    TOOL_NODES,
    TOOL_STATUS_MESSAGES,
    resolve_pending,
)
from ax_rag.shared.config import get_config

# 그래프 노드 이름 (graph.py의 add_node/add_edge와 분기 반환값이 쓰는 값).
# 도구 노드의 이름은 intent 값 그대로라 여기 없다 (tools.TOOL_NODES 키)
NODE_ROUTE = "route"
NODE_DENSE_RETRIEVE = "dense_retrieve"
NODE_BM25_RETRIEVE = "bm25_retrieve"
NODE_FUSE = "fuse"
NODE_RERANK = "rerank"
NODE_GENERATE = "generate"
NODE_VERIFY = "verify"
NODE_FINALIZE = "finalize"
NODE_INCREMENT_RETRY = "increment_retry"
NODE_FALLBACK = "fallback"
NODE_KNOWLEDGE_ANSWER = "knowledge_answer"
# ReAct 루프 전용 노드 (AGENT_MODE=true)
NODE_AGENT = "agent"
NODE_ACT = "act"
NODE_RETRY_SEARCH = "retry_search"
NODE_DIRECT_ANSWER = "direct_answer"
NODE_DEFERRED = "deferred"

# 검색 진행 문구. 재검색은 문구를 달리해 "왜 또 찾나"가 드러나게 한다
# (stage 값은 둘 다 "retrieve" — 계약 불변)
_SEARCH_MESSAGE = "군 내부 문서를 검색하는 중..."
_RESEARCH_MESSAGE = "근거를 더 찾는 중..."
_GENERATE_MESSAGE = "답변을 생성하는 중..."


def _agent_stage_status(state: QueryState) -> tuple[str, str] | None:
    """agent가 정한 다음 행동을 보고 진행 안내를 만든다.

    agent 노드가 끝난 시점에는 행동이 정해졌지만 아직 실행되지 않았다 —
    그래서 "행동을 설명한 뒤 행동한다"는 순서가 그래프 구조상 보장된다.
    """
    action = str(state.get("next_action") or "")
    if action == ACTION_SEARCH:
        first_search = not (state.get("search_calls") or 0)
        return ("retrieve", _SEARCH_MESSAGE if first_search else _RESEARCH_MESSAGE)
    if action == ACTION_FINISH:
        return ("generate", _GENERATE_MESSAGE)
    tool = AGENT_TOOLS.get(action)
    if tool is not None:
        return (
            "tool",
            TOOL_STATUS_MESSAGES.get(tool.intent, DEFAULT_TOOL_STATUS_MESSAGE),
        )
    return None


def _deferred_stage_status(state: QueryState) -> tuple[str, str] | None:
    """확정 답변 뒤에 실행될 지연 도구가 있으면 그 진행을 안내한다."""
    actions = state.get("deferred_actions") or []
    if not actions:
        return None
    tool = AGENT_TOOLS.get(actions[0])
    if tool is None:
        return None
    return ("tool", TOOL_STATUS_MESSAGES.get(tool.intent, DEFAULT_TOOL_STATUS_MESSAGE))


def _next_stage_status(state: QueryState) -> tuple[str, str] | None:
    """실행 큐(pending_intents)의 선두를 보고 다음 단계 안내를 만든다.

    - 다음이 도구 → stage="tool" + 도구별 문구 (TOOL_STATUS_MESSAGES, 레지스트리 기반)
    - 다음이 DOC_SEARCH → stage="retrieve" (문서 검색 시작)
    - 큐 소진 → None (finalize는 즉시 끝나므로 안내 불필요)
    """
    pending = resolve_pending(state)
    if not pending:
        return None
    head = pending[0]
    if head == DOC_SEARCH or head not in TOOL_NODES:
        return ("retrieve", "군 내부 문서를 검색하는 중...")
    if head in TERMINAL_ONLY_TOOLS:
        # 단독 전용 도구(SMALLTALK)는 곧바로 답변 생성이다
        return ("generate", "답변을 생성하는 중...")
    return ("tool", TOOL_STATUS_MESSAGES.get(head, DEFAULT_TOOL_STATUS_MESSAGE))


def status_after_node(node_name: str, state: QueryState) -> tuple[str, str] | None:
    """그래프 노드 완료 시 다음 단계 안내 (stage, message). 안내가 없는 노드는 None.

    프론트가 "검색하는 중..." 같은 진행 상태를 표시할 수 있게 하는
    status 이벤트의 재료다. 완료된 노드를 보고 "이제 시작되는 단계"를 알린다.
    """
    if node_name == NODE_AGENT:
        # ReAct: 정해진 다음 행동을 실행 **전에** 알린다
        return _agent_stage_status(state)
    if node_name == NODE_RETRY_SEARCH:
        # 반려 후 다시 판단하러 가는 중 (다음 안내는 agent가 낸다)
        return ("route", "답변을 다시 검토하는 중...")
    if node_name == NODE_ROUTE:
        return _next_stage_status(state)
    if node_name in TOOL_NODES and node_name not in TERMINAL_ONLY_TOOLS:
        # 계획 실행 중인 도구 완료 → 남은 큐 기준으로 다음 단계 안내
        return _next_stage_status(state)
    if node_name == NODE_FINALIZE:
        # ReAct: 확정 답변 뒤에 실행될 지연 도구(파일 저장 등) 안내
        deferred = _deferred_stage_status(state)
        if deferred is not None:
            return deferred
        # 복합 계획의 후처리 도구(HWP_EXPORT 등)가 남아 있으면 실행 안내
        pending = state.get("pending_intents") or []
        head = pending[0] if pending else None
        if head in TOOL_NODES:
            return ("tool", TOOL_STATUS_MESSAGES.get(head, DEFAULT_TOOL_STATUS_MESSAGE))
        return None
    if node_name == NODE_DIRECT_ANSWER:
        return _deferred_stage_status(state)
    if node_name == NODE_FUSE:
        return ("rerank", "관련 문서를 선별하는 중...")
    if node_name == NODE_RERANK:
        return ("generate", "답변을 생성하는 중...")
    if node_name == NODE_GENERATE:
        return ("verify", "답변이 문서에 근거하는지 검증하는 중...")
    if node_name == NODE_INCREMENT_RETRY:
        return ("generate", "답변을 다시 생성하는 중...")
    return None


def status_event_payload(node_name: str, state: QueryState) -> dict | None:
    """SSE status 이벤트 페이로드. 보낼 것이 없으면 None.

    stage·message는 status_after_node가 정하고, 여기서는 ReAct의 판단 근거
    (thought)와 라운드 번호를 **선택 필드로 얹는다**. 미들웨어가 모르는 필드를
    무시하면 현행과 완전히 같은 동작이 된다 — 그래서 계약을 깨지 않는다
    (docs/react_migration_plan.md §5.2).

    thought는 agent 노드에서만 싣는다. 다른 노드에도 실으면 이미 지나간
    판단이 반복 표시된다.
    """
    status = status_after_node(node_name, state)
    if status is None:
        return None
    stage, message = status
    payload: dict = {"type": "status", "stage": stage, "message": message}

    thought = str(state.get("agent_thought") or "")
    if node_name == NODE_AGENT and thought and get_config().STREAM_THOUGHTS:
        payload["thought"] = thought
        payload["step"] = state.get("agent_steps") or 1
    return payload

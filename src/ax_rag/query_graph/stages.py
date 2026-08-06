"""그래프 노드 이름과 진행 상태(SSE status) 안내 문구.

노드 이름을 API 계층이 문자열로 알고 있으면, 노드 이름을 바꿨을 때
진행 안내가 조용히 사라진다 (테스트도 못 잡는다). 그래서 노드 이름 상수와
"이 노드가 끝나면 다음은 무엇인가"의 판단을 그래프 쪽에 둔다 —
graph.py가 노드를 등록할 때 쓰는 이름과 같은 상수를 여기서 쓴다.

status 이벤트의 stage 값("retrieve", "rerank", ...)은 미들웨어·프론트와의
계약이다 (interfaces.md §5, POST /query 응답 문서). 노드 이름과 별개이며
바꾸면 프론트 표시가 깨진다.
"""

from __future__ import annotations

from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import (
    DEFAULT_TOOL_STATUS_MESSAGE,
    DOC_SEARCH,
    TERMINAL_ONLY_TOOLS,
    TOOL_NODES,
    TOOL_STATUS_MESSAGES,
    resolve_pending,
)

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
    if node_name == NODE_ROUTE:
        return _next_stage_status(state)
    if node_name in TOOL_NODES and node_name not in TERMINAL_ONLY_TOOLS:
        # 계획 실행 중인 도구 완료 → 남은 큐 기준으로 다음 단계 안내
        return _next_stage_status(state)
    if node_name == NODE_FINALIZE:
        # 복합 계획의 후처리 도구(HWP_EXPORT 등)가 남아 있으면 실행 안내
        pending = state.get("pending_intents") or []
        head = pending[0] if pending else None
        if head in TOOL_NODES:
            return ("tool", TOOL_STATUS_MESSAGES.get(head, DEFAULT_TOOL_STATUS_MESSAGE))
        return None
    if node_name == NODE_FUSE:
        return ("rerank", "관련 문서를 선별하는 중...")
    if node_name == NODE_RERANK:
        return ("generate", "답변을 생성하는 중...")
    if node_name == NODE_GENERATE:
        return ("verify", "답변이 문서에 근거하는지 검증하는 중...")
    if node_name == NODE_INCREMENT_RETRY:
        return ("generate", "답변을 다시 생성하는 중...")
    return None

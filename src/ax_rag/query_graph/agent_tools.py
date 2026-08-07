"""ReAct 에이전트의 행동 레지스트리와 관측(observation) 포매팅.

기존 도구 레지스트리(tools.py)를 대체하지 않고 **덧입힌다**: 실행 함수는
tools.TOOL_NODES의 것을 그대로 쓰고, 여기서는 에이전트가 부르는 행동 이름과
실행 시점(phase), 그리고 행동 어휘로 다시 쓴 설명을 얹는다.

설명만 따로 두는 이유는 AgentTool.description 주석에 적었다 — 라우터용
설명에는 intent 이름(DOC_SEARCH 등)이 본문에 섞여 있어 그대로 보여주면
모델이 행동 이름 대신 그 값을 출력한다.

실행 시점(phase):
- inline   — 루프 안에서 즉시 실행하고 결과를 관측으로 돌려준다
- deferred — verify 통과 **후**에 실행한다. "방금 만든 답변"을 입력으로
  쓰는 도구(파일 저장)가 여기 속한다. 검증 실패 시 실행되지 않는다
  (fail-closed: 검증 못 받은 답변을 파일로 만들지 않는다)

관측은 요약본이다. 근거 전문은 상태(retrieved_chunks)에 쌓고 에이전트에게는
발췌만 보여준다 — 스크래치패드가 부풀면 16K 예산이 깨진다
(docs/react_migration_plan.md §4 D2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import TOOL_NODES

# 검색과 종료는 도구가 아니라 루프의 기본 행동이다 (TOOL_NODES에 없다)
ACTION_SEARCH = "search_documents"
ACTION_FINISH = "finish"

# 검색 결과를 합성할 때 쓰는 경로 이름 (graph._compose_final이 읽는 intents 값)
DOC_SEARCH = "DOC_SEARCH"

PHASE_INLINE = "inline"
PHASE_DEFERRED = "deferred"

# 관측 1건의 문자 상한. 이 값 x MAX_AGENT_STEPS가 스크래치패드 예산이다
MAX_OBSERVATION_CHARS = 400
# 근거 1건당 보여줄 발췌 길이 (에이전트는 "무엇이 잡혔는지"만 알면 된다)
_EXCERPT_CHARS = 150
_MAX_EXCERPTS = 3


@dataclass(frozen=True)
class AgentTool:
    """에이전트가 고를 수 있는 행동 하나."""

    name: str  # LLM이 action 값으로 쓰는 이름
    intent: str  # tools.py 레지스트리 키 (합성 순서·진행 문구가 이 값을 쓴다)
    phase: str  # PHASE_INLINE | PHASE_DEFERRED
    node: Callable[[dict], dict]  # 실행 함수 (tools.TOOL_NODES 재사용)
    # 에이전트 프롬프트용 설명. tools.TOOL_DESCRIPTIONS를 그대로 쓰지 않는다 —
    # 그쪽은 라우터용이라 본문에 "DOC_SEARCH", "HWP_EXPORT" 같은 **intent 이름**이
    # 섞여 있어서, 그대로 보여주면 모델이 행동 이름 대신 intent 이름을 출력한다
    # (action="DOC_SEARCH" → 미지의 행동 → 검색이 통째로 사라진다).
    # 분류 기준 자체는 같게 유지하고 어휘만 행동 이름으로 옮긴다
    description: str


# 행동 이름 → 명세. SMALLTALK은 여기 없다 — 에이전트가 도구 없이 finish하면
# direct_answer 노드가 그 역할을 한다 (도구로 두면 "잡담을 실행"이라는
# 어색한 행동이 생기고, 업무 답변과 합성될 위험도 생긴다)
AGENT_TOOLS: dict[str, AgentTool] = {
    "discharge_days": AgentTool(
        name="discharge_days",
        intent="DISCHARGE_DAYS",
        phase=PHASE_INLINE,
        node=TOOL_NODES["DISCHARGE_DAYS"],
        description=(
            "전역일을 알려주거나 전역까지 남은 날짜를 묻는 발화 전부 "
            '(예: "내 전역일은 2026년 12월 1일이야", "전역까지 며칠 남았어?"). '
            f"단, 전역 절차·규정 질문은 {ACTION_SEARCH}"
        ),
    ),
    "draft_document": AgentTool(
        name="draft_document",
        intent="HWP_DRAFT",
        phase=PHASE_INLINE,
        node=TOOL_NODES["HWP_DRAFT"],
        description=(
            "사용자가 요청에 담아 준 내용으로 **새 문서 초안**(공문·보고서 등)을 "
            '작성해 파일로 만든다 (예: "이 내용으로 공문 초본 잡아서 파일로 만들어줘"). '
            "기존 답변을 그대로 저장하는 것은 export_hwp"
        ),
    ),
    "export_hwp": AgentTool(
        name="export_hwp",
        intent="HWP_EXPORT",
        phase=PHASE_DEFERRED,
        node=TOOL_NODES["HWP_EXPORT"],
        description=(
            "직전 답변이나 지금 만들 답변을 **있는 그대로** 한글 문서 파일로 저장한다. "
            "'한글로 저장'뿐 아니라 '문서로 만들어줘', '문서화해줘', '파일로 뽑아줘'도 "
            "포함한다. 검색이 함께 필요하면 "
            f"{ACTION_SEARCH}를 먼저 하고 그 다음 라운드에서 이 행동을 고른다 "
            '(예: "휴가 규정 찾아서 파일로 만들어줘"). '
            "단, 문서 작성 방법·절차를 묻는 질문은 검색 대상이다"
        ),
    ),
}

# intent 값 → 행동 이름 (강제 지정·결정적 매처가 intent로 들어온다)
INTENT_TO_ACTION: dict[str, str] = {tool.intent: name for name, tool in AGENT_TOOLS.items()}

# 7B 허용 오차: 모델이 행동 이름 대신 intent 이름이나 흔한 변형을 낼 때의 보정.
# 라우터가 문자열 intents·단수형 intent를 보정해 받는 것과 같은 취지다 —
# 형태가 살짝 어긋났다고 검색을 통째로 잃는 것이 훨씬 큰 손해다
ACTION_ALIASES: dict[str, str] = {
    "doc_search": ACTION_SEARCH,
    "search": ACTION_SEARCH,
    "retrieve": ACTION_SEARCH,
    "hwp_export": "export_hwp",
    "hwp_draft": "draft_document",
    "discharge": "discharge_days",
    # 잡담·직접 답변은 행동이 아니라 "도구 없이 종료"다
    "smalltalk": ACTION_FINISH,
    "answer": ACTION_FINISH,
    "none": ACTION_FINISH,
}


def action_names() -> tuple[str, ...]:
    """유효한 행동 이름 전체 (검색·종료 포함)."""
    return (ACTION_SEARCH, *AGENT_TOOLS, ACTION_FINISH)


def resolve_action(raw: str) -> str:
    """모델이 낸 행동 이름을 유효한 값으로 보정한다. 알 수 없으면 빈 문자열."""
    name = str(raw or "").strip().lower()
    name = ACTION_ALIASES.get(name, name)
    return name if name in action_names() else ""


def action_guide() -> str:
    """에이전트 시스템 프롬프트의 행동 목록 (레지스트리에서 생성).

    도구를 추가하면 프롬프트가 자동으로 따라온다 — 라우터 프롬프트가
    TOOL_DESCRIPTIONS에서 생성되던 것과 같은 방식이다.
    """
    lines = [
        f"- {ACTION_SEARCH}: 군 내부 문서를 검색한다. query에 검색어를 넣는다 "
        "(규정·절차·수치를 묻는 업무 질문은 반드시 이것부터)",
    ]
    lines += [f"- {tool.name}: {tool.description}" for tool in AGENT_TOOLS.values()]
    lines.append(f"- {ACTION_FINISH}: 더 할 행동이 없다. 모은 근거로 답변 작성 단계로 넘어간다")
    return "\n".join(lines)


def _wrap(body: str) -> str:
    """관측을 delimiter로 감싼다 (인젝션 방어면 — 프롬프트가 '데이터로만 취급'을 지시).

    상한을 넘으면 자른다. 에이전트는 "무엇이 잡혔는지"만 알면 되고,
    답변 작성에 쓰이는 근거 전문은 상태에 따로 쌓여 있다.
    """
    trimmed = body.strip()
    if len(trimmed) > MAX_OBSERVATION_CHARS:
        trimmed = trimmed[:MAX_OBSERVATION_CHARS].rstrip() + "…"
    return f"<observation>\n{trimmed}\n</observation>"


def format_search_observation(query: str, chunks: list[dict], total: int) -> str:
    """검색 결과 요약 관측. 문서명과 짧은 발췌만 담는다.

    total은 이번 검색 뒤 **누적** 근거 건수다 — 에이전트가 "더 찾을지"를
    판단하는 기준이 누적치여야 하기 때문이다.
    """
    if not chunks:
        return _wrap(f'검색어 "{query}": 관련 근거를 찾지 못했다 (0건). 누적 근거 {total}건')
    names = sorted(
        {str(chunk.get("source_doc") or "") for chunk in chunks if chunk.get("source_doc")}
    )
    excerpts = [
        str(chunk.get("text") or "").strip().replace("\n", " ")[:_EXCERPT_CHARS]
        for chunk in chunks[:_MAX_EXCERPTS]
    ]
    body = (
        f'검색어 "{query}": 근거 {len(chunks)}건 (누적 {total}건)\n'
        f"문서: {', '.join(names)}\n"
        + "\n".join(f"- {excerpt}…" for excerpt in excerpts if excerpt)
    )
    return _wrap(body)


def format_tool_observation(action: str, answer: str) -> str:
    """도구 실행 결과 관측. 도구 답변은 이미 사용자 답변에 합성되므로 요약만 싣는다."""
    summary = str(answer or "").strip().replace("\n", " ")
    return _wrap(f"{action} 실행 완료: {summary}")


def format_deferred_observation(action: str) -> str:
    """지연 도구 예약 관측 — 아직 실행하지 않았음을 분명히 한다.

    "예약"이라고 알려 주지 않으면 에이전트가 파일이 이미 만들어진 줄 알고
    같은 행동을 반복한다.
    """
    return _wrap(f"{action}: 답변이 확정된 뒤 실행하도록 예약했다 (아직 실행 전)")


def format_duplicate_observation(query: str) -> str:
    """중복 검색어 관측 — 같은 검색을 반복해도 결과가 같음을 알린다."""
    return _wrap(f'검색어 "{query}"로는 이미 검색했다. 같은 결과가 나오므로 더 찾지 않는다')


def format_verify_observation(reason: str) -> str:
    """검증 반려 사유 관측 — 재검색의 근거가 된다 (graph.retry_search)."""
    body = (
        f"직전 답변이 근거 검증에서 반려되었다. 사유: {reason}\n"
        "근거가 부족한 부분을 다른 검색어로 다시 찾거나, 더 찾을 것이 없으면 finish 한다."
    )
    return _wrap(body)


def scratchpad_text(state: QueryState) -> str:
    """스크래치패드를 에이전트 프롬프트용 텍스트로 만든다. 비어 있으면 빈 문자열."""
    entries = state.get("agent_scratchpad") or []
    if not entries:
        return ""
    blocks = []
    for index, entry in enumerate(entries, start=1):
        action = str(entry.get("action") or "")
        observation = str(entry.get("observation") or "")
        blocks.append(f"[{index}] 행동: {action}\n{observation}")
    return "\n\n".join(blocks)

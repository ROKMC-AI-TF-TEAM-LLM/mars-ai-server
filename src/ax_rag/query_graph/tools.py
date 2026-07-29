"""도구 레지스트리: intent 값 → 처리 노드 (code_guide.md §12 패턴 B의 구현).

커스텀 도구 추가 절차:
1) nodes/<도구>.py 작성 — smalltalk과 동일 계약: state를 받아
   tool_contract.tool_answer(답변)의 반환값을 그대로 돌려준다
   (반환 dict를 손으로 적지 말 것 — grounded 누락 시 검증 실패 답변에
   출처가 붙는다. 파일을 만드는 도구는 generated_files 인자를 함께 넘긴다)
2) 아래 TOOL_NODES에 노드 등록 + TOOL_DESCRIPTIONS에 분류 기준 한 줄
3) (선택) 결정적으로 감지 가능한 도구면 TOOL_MATCHERS에 매처 등록 —
   LLM 분류보다 먼저 코드로 판정해 오분류·지연을 없앤다
4) (선택) 복합 계획에 섞이면 안 되는 도구면 TERMINAL_ONLY_TOOLS에 추가
5) (선택) "방금 만든 답변"을 입력으로 쓰는 도구면 POST_SEARCH_TOOLS에 추가
   — 검색 파이프라인 뒤(finalize 후)에 실행된다. 검증 실패 시 미실행
6) (선택) 실행 중 프론트에 보여줄 문구를 TOOL_STATUS_MESSAGES에 등록
   (미등록이면 기본 문구 "요청을 처리하는 중...")

이것만으로 그래프 배선(graph.py — 복합 계획 편입 포함), 라우터 분류
항목(router.py), 강제 선택 허용값(main.normalize_tool)이 전부 자동 반영된다.

DOC_SEARCH는 도구가 아니라 기본 파이프라인(검색→생성→검증)이므로
TOOL_NODES에 넣지 않는다. 도메인 한정 검색(교범/훈령 모드)도 도구가 아니라
요청의 domain 필드로 처리한다 (interfaces.md §5).
"""

from __future__ import annotations

from collections.abc import Callable

from ax_rag.query_graph.nodes.discharge_days import discharge_days, is_discharge_request
from ax_rag.query_graph.nodes.hwp_draft import hwp_draft
from ax_rag.query_graph.nodes.hwp_export import hwp_export, is_hwp_export_request
from ax_rag.query_graph.nodes.smalltalk import smalltalk
from ax_rag.query_graph.state import QueryState

# 기본 경로: 문서 검색 파이프라인 (도구 아님)
DOC_SEARCH = "DOC_SEARCH"

# intent 값 → 노드 함수. 그래프 노드 이름 = intent 값
TOOL_NODES: dict[str, Callable[[dict], dict]] = {
    "SMALLTALK": smalltalk,
    "DISCHARGE_DAYS": discharge_days,
    "HWP_EXPORT": hwp_export,
    "HWP_DRAFT": hwp_draft,
}

# 라우터 프롬프트에 들어가는 분류 기준 (intent 값 → 한 줄 설명)
TOOL_DESCRIPTIONS: dict[str, str] = {
    DOC_SEARCH: "군 내부 문서 검색이 필요한 업무·규정·행정 질문 (애매하면 이것)",
    "SMALLTALK": "인사, 자기소개, 감사, 잡담, 챗봇 자신에 대한 질문 "
    '(예: "안녕", "너 뭐 할 수 있어?")',
    "DISCHARGE_DAYS": "전역일을 알려주거나 전역까지 남은 날짜를 묻는 발화 전부 "
    '(예: "내 전역일은 2026년 12월 1일이야", "전역까지 며칠 남았어?", "전역 D-day 알려줘". '
    "단, 전역 절차·규정 질문은 DOC_SEARCH)",
    "HWP_EXPORT": "직전 답변이나 검색 결과를 **있는 그대로** 문서 파일(한글 HWP)로 "
    "저장·생성해 달라는 요청. '한글로 저장'뿐 아니라 '문서로 만들어줘', "
    "'문서화해줘', '파일로 뽑아줘'처럼 표현이 '한글'을 안 붙여도 포함한다. "
    "검색이 필요하면 DOC_SEARCH와 함께 나열한다 "
    '(예: "이 답변 저장해줘" → HWP_EXPORT; '
    '"해병대 조사해서 문서로 만들어줘"·"휴가 규정 찾아서 파일로 만들어줘" → '
    "DOC_SEARCH, HWP_EXPORT). 단, 문서 작성 방법·절차를 묻는 질문은 DOC_SEARCH)",
    "HWP_DRAFT": "사용자가 요청에 담아 준 내용으로 **새 문서 초안**(공문·보고서 등)을 "
    '작성해 파일로 만들어 달라는 요청 (예: "이 내용으로 공문 초본 잡아서 파일로 생성해줘". '
    "기존 답변을 그대로 저장하는 건 HWP_EXPORT)",
}

# generate/verify의 "도구가 처리함" 안내문(tool_handled_note)용 짧은 라벨.
# TOOL_DESCRIPTIONS(라우터용)를 그대로 쓰면 안 된다 — 분류용 예시 문구
# ("해병대 조사해서 문서로 만들어줘")를 7B 검증기가 답변 내용으로 착각해
# grounded=false 오탐을 낸다 (실측: 예시 주제가 실제 질문과 겹칠 때).
# 예시·질문형 문구 없이 요청 유형만 서술한다
TOOL_HANDLED_LABELS: dict[str, str] = {
    "SMALLTALK": "잡담·인사 응대",
    "DISCHARGE_DAYS": "전역일·남은 날짜 계산",
    "HWP_EXPORT": "답변·검색 결과를 한글(HWPX) 문서 파일로 저장",
    "HWP_DRAFT": "사용자 제공 내용으로 문서 초안 작성 및 파일 생성",
}


# 결정적 매처: LLM 분류 전에 코드로 판정한다 (intent 값 → 판정 함수).
# 매치되면 라우터가 LLM 호출 없이 즉시 해당 도구로 보낸다 — 빠르고 오분류 없음
TOOL_MATCHERS: dict[str, Callable[[str], bool]] = {
    "DISCHARGE_DAYS": is_discharge_request,
    "HWP_EXPORT": is_hwp_export_request,
}


# 요청의 tool 필드로 강제 지정을 허용하는 도구 화이트리스트.
# SMALLTALK은 제외: 강제 잡담 경로로 업무 질문이 들어오면 verify 밖에서
# 모델이 규정을 지어내는 것을 실측 — 프롬프트로 막히지 않아 구조적으로 차단한다.
# 강제를 전제로 설계됐거나 강제로 들어와도 안전한 도구만 등록할 것.
# - HWP_EXPORT: 결정적 코드 도구 — 프론트 "답변 저장" 버튼이 강제 지정으로 쓴다
# - HWP_DRAFT: LLM 생성이지만 사용자 제공 내용의 형식화라 강제 유입이 안전
#   (규정·수치를 물어도 문서 초안으로 정리할 뿐 사실을 답하지 않음)
FORCIBLE_TOOLS: frozenset[str] = frozenset({"DISCHARGE_DAYS", "HWP_EXPORT", "HWP_DRAFT"})


# 도구 실행 직전 SSE status로 내보내는 진행 안내 문구 (main._status_after_node).
# 미등록 도구는 DEFAULT_TOOL_STATUS_MESSAGE를 쓴다
TOOL_STATUS_MESSAGES: dict[str, str] = {
    "DISCHARGE_DAYS": "전역일을 계산하는 중...",
    "HWP_EXPORT": "한글 문서를 만드는 중...",
    "HWP_DRAFT": "문서 초안을 작성하는 중...",
}
DEFAULT_TOOL_STATUS_MESSAGE = "요청을 처리하는 중..."


# 복합 계획(여러 경로 합성)에 섞일 수 없는 단독 전용 도구.
# - SMALLTALK: verify 밖 자유 생성이라 업무 답변과 한 응답으로 합성하지 않는다
# - HWP_DRAFT: 사용자 제공 내용의 초안 작성은 자체 완결 작업이다 (검색 결과
#   저장이 필요한 복합 요청은 HWP_EXPORT가 후처리로 담당)
TERMINAL_ONLY_TOOLS: frozenset[str] = frozenset({"SMALLTALK", "HWP_DRAFT"})


# 후처리 도구: 검색 파이프라인(verify·finalize) **뒤에** 실행된다.
# HWP_EXPORT처럼 "방금 만든 답변"을 입력으로 쓰는 도구가 여기 속한다 —
# "휴가 규정 찾아서 한글 파일로 저장해줘" 복합 질문에서 검증 통과한 답변을
# 파일로 만든다. 검증 실패(fallback) 시에는 실행되지 않는다 (실패 답변을
# 파일로 만들지 않는다 — fail-closed)
POST_SEARCH_TOOLS: frozenset[str] = frozenset({"HWP_EXPORT"})


def valid_intents() -> tuple[str, ...]:
    """허용되는 intent 값 전체 (기본 경로 + 등록된 도구)."""
    return (DOC_SEARCH, *TOOL_NODES)


def format_handled_note(state: QueryState, template: str) -> str:
    """도구가 담당한 요청 유형을 안내하는 꼬리 프롬프트를 만든다 (generate/verify 공용).

    이미 실행된 전처리 도구(tool_answers)뿐 아니라 **아직 실행 전인 후처리
    도구**(계획 속 POST_SEARCH_TOOLS — 파일 저장 등)도 포함한다:
    - generate: 빠뜨리면 "그 기능은 제공하지 않는다" 같은 잘못된 사족을 붙인다 (실측)
    - verify: 빠뜨리면 답변이 도구 몫을 안 다뤘다고 grounded=false 오탐을 낸다 (실측)

    도구 답변의 수치는 넣지 않는다 — 초안에 섞이면 rule_based_verify가
    "근거에 없는 수치"로 오탐한다. 라우터용 상세 설명(TOOL_DESCRIPTIONS)이
    아니라 예시 없는 짧은 라벨(TOOL_HANDLED_LABELS)을 쓰는 이유도 같다:
    분류 예시 문구가 안내문에 실리면 7B가 답변 내용으로 착각한다 (실측).

    담당 도구가 없으면 빈 문자열 (안내문 자체를 붙이지 않는다).
    """
    handled = [item.get("intent") for item in (state.get("tool_answers") or [])]
    handled += [
        name
        for name in (state.get("intents") or [])
        if name in POST_SEARCH_TOOLS and name not in handled
    ]
    if not handled:
        return ""
    lines = "\n".join(f"- {TOOL_HANDLED_LABELS.get(name, name)}" for name in handled if name)
    return template.format(handled=lines)


def plan_of(state: QueryState) -> list[str]:
    """상태에서 처리 계획(intents)을 읽는다. 구형 단일 intent 상태도 허용한다 (방어적).

    route가 항상 intents를 채우지만, 계획 이전 형태의 상태(대표 intent만 있는
    입력·테스트 픽스처)로도 그래프가 진행되게 한다. 미지의 intent 값은
    DOC_SEARCH로 떨어뜨린다 — 존재하지 않는 노드로 분기하지 않기 위해서다.
    """
    plan = state.get("intents")
    if plan:
        return list(plan)
    intent = state.get("intent") or DOC_SEARCH
    return [intent if intent in valid_intents() else DOC_SEARCH]


def resolve_pending(state: QueryState) -> list[str]:
    """남은 실행 큐(pending_intents)를 읽는다. 없으면 계획에서 재구성한다 (방어적).

    그래프 분기(graph.next_step)와 진행 상태 안내(stages.status_after_node)가
    같은 큐 해석을 쓰도록 한 곳에 모은다.
    """
    pending = state.get("pending_intents")
    if pending is None:  # 구형 상태: 계획에서 실행 큐를 재구성
        return execution_queue(plan_of(state))
    return list(pending)


def execution_queue(plan: list[str]) -> list[str]:
    """계획(intents)을 실행 순서로 바꾼다:
    [전처리 도구들(계획 순서)] → DOC_SEARCH → [후처리 도구들(계획 순서)].

    전처리 도구(D-day 등)는 검색과 독립이라 앞에, 후처리 도구(HWP_EXPORT 등,
    POST_SEARCH_TOOLS)는 방금 만든 답변을 입력으로 쓰므로 검색 파이프라인
    뒤에 둔다. 최종 답변의 합성 순서는 실행 순서가 아니라 계획(intents)
    순서를 따른다 (graph._compose_final).
    """
    pre = [n for n in plan if n != DOC_SEARCH and n not in POST_SEARCH_TOOLS]
    post = [n for n in plan if n in POST_SEARCH_TOOLS]
    middle = [DOC_SEARCH] if DOC_SEARCH in plan else []
    return [*pre, *middle, *post]

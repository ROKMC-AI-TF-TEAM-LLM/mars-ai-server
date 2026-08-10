"""도구 레지스트리: intent 값 → 처리 노드 + 미들웨어 계약 어휘.

이 파일은 **intent 이름 체계**를 소유한다. 그 이름이 API 계약(POST /query의
tool 필드, GET /capabilities)에 그대로 노출되기 때문이다. 에이전트가 부르는
**행동 이름**은 별개이며 agent_tools.py가 소유한다 (intent ↔ 행동 매핑도 거기).

커스텀 도구 추가 절차:
1) nodes/<도구>.py 작성 — state를 받아 tool_contract.tool_answer()의 반환값을
   그대로 돌려준다 (dict를 손으로 적지 말 것: grounded 누락 시 검증 실패
   답변에 출처가 붙는다)
2) TOOL_NODES + TOOL_DESCRIPTIONS(한 줄 설명) + TOOL_HANDLED_LABELS(예시 없는 라벨)
3) agent_tools.AGENT_TOOLS에 행동 등록. 실행 시점은 POST_SEARCH_TOOLS 포함
   여부로 자동 결정된다
4) (선택) TOOL_MATCHERS — 코드로 확정 가능한 도구면 LLM 판단보다 먼저 잡는다
5) (선택) POST_SEARCH_TOOLS — "방금 만든 답변"을 입력으로 쓰는 도구
6) (선택) TOOL_STATUS_MESSAGES — 실행 중 진행 문구

이것만으로 그래프 배선, 에이전트 행동 목록, 진행 문구, 강제 선택 허용값,
/capabilities 응답이 전부 자동 반영된다.

DOC_SEARCH는 도구가 아니라 검색 경로 이름이라 TOOL_NODES에 넣지 않는다.
도메인 한정도 도구가 아니라 요청의 domain 필드로 처리한다 (interfaces.md §5).
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
    "SMALLTALK": "인사, 자기소개, 감사, 잡담, 챗봇 자신에 대한 질문, "
    "그리고 **규정·제도를 묻지 않는** 개인 상황 토로·고민 상담·감정 표현 "
    '(예: "안녕", "너 뭐 할 수 있어?", "나 요즘 힘들어", '
    '"난 해병대 장교인데 고민이 있어"). '
    "단, 구체적인 규정·절차·수치를 물으면 DOC_SEARCH",
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

# generate/verify 안내문용 짧은 라벨. TOOL_DESCRIPTIONS(라우터용)를 쓰면 안 된다 —
# 분류용 예시 문구를 검증기가 답변 내용으로 착각해 오탐을 낸다. 예시 없이 유형만 서술할 것
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


# 요청의 tool 필드로 강제 지정을 허용하는 화이트리스트.
# ⚠️ SMALLTALK 제외 — 강제 잡담 경로로 업무 질문이 들어오면 verify 밖에서
# 모델이 규정을 지어내는 것을 실측했고, 프롬프트로 막히지 않아 구조적으로 차단한다.
# 강제로 들어와도 안전한 도구(결정적 코드 도구, 사용자 제공 내용의 형식화)만 등록할 것
FORCIBLE_TOOLS: frozenset[str] = frozenset({"DISCHARGE_DAYS", "HWP_EXPORT", "HWP_DRAFT"})


# 도구 실행 직전 SSE status로 내보내는 진행 안내 문구 (main._status_after_node).
# 미등록 도구는 DEFAULT_TOOL_STATUS_MESSAGE를 쓴다
TOOL_STATUS_MESSAGES: dict[str, str] = {
    "DISCHARGE_DAYS": "전역일을 계산하는 중...",
    "HWP_EXPORT": "한글 문서를 만드는 중...",
    "HWP_DRAFT": "문서 초안을 작성하는 중...",
}
DEFAULT_TOOL_STATUS_MESSAGE = "요청을 처리하는 중..."


# 지연 실행 도구: 검증(verify)·확정(finalize) **뒤에** 실행된다.
# HWP_EXPORT처럼 "방금 만든 답변"을 입력으로 쓰는 도구가 여기 속한다 —
# "휴가 규정 찾아서 한글 파일로 저장해줘" 요청에서 검증 통과한 답변을
# 파일로 만든다. 검증 실패(fallback·knowledge) 시에는 실행되지 않는다
# (실패 답변을 파일로 만들지 않는다 — fail-closed).
#
# 이 집합이 **단일 출처**다: agent_tools가 여기서 행동의 실행 시점(phase)을
# 끌어가고, 아래 format_handled_note가 generate·verify 안내문에 쓴다.
POST_SEARCH_TOOLS: frozenset[str] = frozenset({"HWP_EXPORT"})


def format_handled_note(state: QueryState, template: str) -> str:
    """도구가 담당한 요청 유형을 안내하는 꼬리 프롬프트 (generate/verify 공용).

    아직 실행 전인 지연 도구도 포함한다 — 빠뜨리면 generate가 "그 기능은
    제공하지 않는다"는 사족을 붙이고, verify가 도구 몫을 안 다뤘다고 오탐을 낸다.

    도구 답변의 수치와 분류 예시 문구는 넣지 않는다. 검증기가 답변 내용으로
    착각해 근거 밖 수치라고 오판한다.
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

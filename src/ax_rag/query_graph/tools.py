"""도구 레지스트리: intent 값 → 처리 노드 (code_guide.md §12 패턴 B의 구현).

커스텀 도구 추가 절차:
1) nodes/<도구>.py 작성 — smalltalk과 동일 계약:
   state를 받아 {"final_answer", "grounded": False, "retrieved_chunks": []} 반환
2) 아래 TOOL_NODES에 노드 등록 + TOOL_DESCRIPTIONS에 분류 기준 한 줄
3) (선택) 결정적으로 감지 가능한 도구면 TOOL_MATCHERS에 매처 등록 —
   LLM 분류보다 먼저 코드로 판정해 오분류·지연을 없앤다

이것만으로 그래프 배선(graph.py), 라우터 분류 항목(router.py),
강제 선택 허용값(main.normalize_tool)이 전부 자동 반영된다.

DOC_SEARCH는 도구가 아니라 기본 파이프라인(검색→생성→검증)이므로
TOOL_NODES에 넣지 않는다. 도메인 한정 검색(교범/훈령 모드)도 도구가 아니라
요청의 domain 필드로 처리한다 (interfaces.md §5).
"""

from __future__ import annotations

from collections.abc import Callable

from ax_rag.query_graph.nodes.discharge_days import discharge_days, is_discharge_request
from ax_rag.query_graph.nodes.smalltalk import smalltalk

# 기본 경로: 문서 검색 파이프라인 (도구 아님)
DOC_SEARCH = "DOC_SEARCH"

# intent 값 → 노드 함수. 그래프 노드 이름 = intent 값
TOOL_NODES: dict[str, Callable[[dict], dict]] = {
    "SMALLTALK": smalltalk,
    "DISCHARGE_DAYS": discharge_days,
}

# 라우터 프롬프트에 들어가는 분류 기준 (intent 값 → 한 줄 설명)
TOOL_DESCRIPTIONS: dict[str, str] = {
    DOC_SEARCH: "군 내부 문서 검색이 필요한 업무·규정·행정 질문 (애매하면 이것)",
    "SMALLTALK": "인사, 자기소개, 감사, 잡담, 챗봇 자신에 대한 질문 "
    '(예: "안녕", "너 뭐 할 수 있어?")',
    "DISCHARGE_DAYS": "전역일을 알려주거나 전역까지 남은 날짜를 묻는 발화 전부 "
    '(예: "내 전역일은 2026년 12월 1일이야", "전역까지 며칠 남았어?", "전역 D-day 알려줘". '
    "단, 전역 절차·규정 질문은 DOC_SEARCH)",
}


# 결정적 매처: LLM 분류 전에 코드로 판정한다 (intent 값 → 판정 함수).
# 매치되면 라우터가 LLM 호출 없이 즉시 해당 도구로 보낸다 — 빠르고 오분류 없음
TOOL_MATCHERS: dict[str, Callable[[str], bool]] = {
    "DISCHARGE_DAYS": is_discharge_request,
}


# 요청의 tool 필드로 강제 지정을 허용하는 도구 화이트리스트.
# SMALLTALK은 제외: 강제 잡담 경로로 업무 질문이 들어오면 verify 밖에서
# 모델이 규정을 지어내는 것을 실측 — 프롬프트로 막히지 않아 구조적으로 차단한다.
# 강제를 전제로 설계됐거나 결정적 코드로만 답하는 도구만 여기에 등록할 것
FORCIBLE_TOOLS: frozenset[str] = frozenset({"DISCHARGE_DAYS"})


def valid_intents() -> tuple[str, ...]:
    """허용되는 intent 값 전체 (기본 경로 + 등록된 도구)."""
    return (DOC_SEARCH, *TOOL_NODES)

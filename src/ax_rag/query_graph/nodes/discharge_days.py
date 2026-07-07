"""discharge_days 도구 노드: 전역일까지 남은 날짜 계산 (테스트용 첫 커스텀 도구).

계산은 LLM이 아니라 코드로 한다 — 날짜 연산을 모델에 맡기면 지어낸 숫자가
나올 수 있고, 이 경로는 verify 밖이므로 결정적 코드만 허용한다.
날짜는 질문에서 먼저 찾고, 없으면 대화 이력의 사용자 발화에서 찾는다
(최근 턴 우선).
"""

from __future__ import annotations

import re
from datetime import date

from ax_rag.query_graph.state import QueryState
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 2026-12-01 / 2026.12.1 / 2026년 12월 1일
_FULL_DATE_RE = re.compile(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*일?")
# 12월 1일 (연도 생략 — 오늘 이후가 되도록 올해/내년 선택)
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")

NO_DATE_ANSWER = (
    "전역일을 찾지 못했습니다. 전역 날짜를 함께 알려주시면 남은 날짜를 계산해 드립니다. "
    '예: "전역일이 2026년 12월 1일인데 며칠 남았어?"'
)

# 결정적 매처용: "며칠/얼마나 남았", "D-day/디데이" 류의 잔여일 표현
_REMAIN_RE = re.compile(r"(며칠|얼마나?)\s*남|남았|남은\s*(날|일수|기간)|[dD]\s*-?\s*[dD]ay|디데이")
# 전역일을 자기 것으로 언급하는 표현 ("전역 절차 알려줘" 같은 규정 질문과 구분)
_MENTION_RE = re.compile(r"전역일|전역\s*날짜|전역해|전역함|전역이야|전역입니다|전역인데|에\s*전역")


def is_discharge_request(question: str, today: date | None = None) -> bool:
    """LLM 없이 판정하는 결정적 매처 (tools.TOOL_MATCHERS 등록용).

    - "전역" + 잔여일 표현("며칠 남았", "D-day") → True
    - 전역일 언급 표현 + 파싱 가능한 날짜 → True
      ("내 전역일은 12월 1일이야"처럼 계산해 달라는 말이 없어도 동작)
    - "전역 절차 알려줘" 같은 규정 질문 → False (문서 검색 경로 유지)
    """
    if "전역" not in question:
        return False
    if _REMAIN_RE.search(question):
        return True
    resolved_today = today or date.today()
    return bool(_MENTION_RE.search(question) and _parse_discharge_date(question, resolved_today))


def _parse_discharge_date(text: str, today: date) -> date | None:
    """텍스트에서 전역일을 추출한다. 잘못된 날짜(13월 등)는 무시."""
    match = _FULL_DATE_RE.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    match = _MONTH_DAY_RE.search(text)
    if match:
        try:
            candidate = date(today.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            return None
        # 연도 생략 시 이미 지난 날짜면 내년으로 해석
        return candidate if candidate >= today else candidate.replace(year=today.year + 1)
    return None


def build_answer(question: str, history: list[dict], today: date) -> str:
    """전역일을 찾아 남은 날짜 답변을 만든다 (순수 함수 — 테스트 용이)."""
    texts = [question] + [
        message.get("content", "")
        for message in reversed(history or [])
        if message.get("role") == "user"
    ]
    discharge = next(
        (found for text in texts if (found := _parse_discharge_date(text, today))), None
    )
    if discharge is None:
        return NO_DATE_ANSWER

    remaining = (discharge - today).days
    formatted = f"{discharge.year}년 {discharge.month}월 {discharge.day}일"
    if remaining > 0:
        return (
            f"전역일 {formatted}까지 D-{remaining}, {remaining}일 남았습니다. "
            "남은 기간도 건강하게 보내시길 바랍니다!"
        )
    if remaining == 0:
        return f"오늘({formatted})이 바로 전역일입니다. 전역을 진심으로 축하드립니다!"
    return f"전역일({formatted})이 이미 {-remaining}일 지났습니다. 전역을 축하드립니다!"


def discharge_days(state: QueryState) -> dict:
    """전역일 D-day 계산. 문서 근거를 주장하지 않으므로 grounded=False."""
    answer = build_answer(state["question"], state.get("conversation_history") or [], date.today())
    logger.info("전역일 계산: %s", answer[:60])
    return {"final_answer": answer, "grounded": False, "retrieved_chunks": []}

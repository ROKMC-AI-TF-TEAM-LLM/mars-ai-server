"""discharge_days 도구 유닛 테스트 — 날짜 추출·계산은 전부 코드(결정적)."""

from __future__ import annotations

from datetime import date

from ax_rag.query_graph.nodes.discharge_days import (
    NO_DATE_ANSWER,
    build_answer,
    discharge_days,
)

_TODAY = date(2026, 7, 7)


def test_한글_날짜_형식_D_day_계산() -> None:
    answer = build_answer("전역일이 2026년 12월 1일인데 며칠 남았어?", [], _TODAY)
    assert "D-147" in answer
    assert "147일 남았습니다" in answer


def test_ISO_날짜_형식도_지원() -> None:
    answer = build_answer("전역: 2027-03-15", [], _TODAY)
    assert "2027년 3월 15일" in answer
    assert "D-251" in answer


def test_연도_생략_시_지난_날짜는_내년으로() -> None:
    answer = build_answer("3월 1일에 전역해. 며칠 남았지?", [], _TODAY)
    assert "2027년 3월 1일" in answer  # 올해 3/1은 지났으므로 내년


def test_오늘이_전역일() -> None:
    answer = build_answer("전역일 2026년 7월 7일!", [], _TODAY)
    assert "오늘" in answer and "축하" in answer


def test_이미_지난_전역일() -> None:
    answer = build_answer("2026년 1월 1일에 전역했는데?", [], _TODAY)
    assert "지났습니다" in answer


def test_날짜가_없으면_안내한다() -> None:
    assert build_answer("전역까지 며칠 남았어?", [], _TODAY) == NO_DATE_ANSWER


def test_잘못된_날짜는_무시한다() -> None:
    assert build_answer("전역일은 2026년 13월 45일이야", [], _TODAY) == NO_DATE_ANSWER


def test_이력의_사용자_발화에서_날짜를_찾는다() -> None:
    history = [
        {"role": "user", "content": "내 전역일은 2026년 12월 1일이야"},
        {"role": "assistant", "content": "네, 기억하겠습니다."},
    ]
    answer = build_answer("며칠 남았어?", history, _TODAY)
    assert "D-147" in answer


def test_챗봇_발화의_날짜는_무시한다() -> None:
    history = [{"role": "assistant", "content": "예시: 2026년 12월 1일"}]
    assert build_answer("며칠 남았어?", history, _TODAY) == NO_DATE_ANSWER


def test_노드_계약_grounded_False() -> None:
    result = discharge_days({"question": "전역일이 2099년 1월 1일이야, 며칠 남았어?"})
    assert result["grounded"] is False  # 문서 근거 미주장 → sources 미노출
    assert result["retrieved_chunks"] == []
    assert "D-" in result["final_answer"]

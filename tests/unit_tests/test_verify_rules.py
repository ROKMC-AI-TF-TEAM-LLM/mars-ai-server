"""rule_based_verify 유닛 테스트 — 숫자/날짜/문서명 검출 (roadmap 4단계 DoD)."""

from __future__ import annotations

from ax_rag.query_graph.nodes.verify import rule_based_verify

_CHUNKS = [
    {
        "text": (
            "[휴가규정.pdf > 연차] 연차휴가는 매년 15일이 부여되며 " "최대 5일까지 이월할 수 있다."
        ),
        "source_doc": "휴가규정.pdf",
    },
    {
        "text": (
            "[휴가규정.pdf > 육아휴직] 육아휴직은 2026년 1월 개정 기준 "
            "자녀 1명당 최대 1년이다. 육아휴직 급여 상한은 월 1,500,000원이다."
        ),
        "source_doc": "휴가규정.pdf",
    },
]


def test_근거에_있는_수치만_쓰면_통과한다() -> None:
    ok, reason = rule_based_verify("연차휴가는 매년 15일이며 최대 5일까지 이월됩니다.", _CHUNKS)
    assert ok, reason


def test_근거에_없는_수치는_실패한다() -> None:
    ok, reason = rule_based_verify("연차휴가는 매년 25일 부여됩니다.", _CHUNKS)
    assert not ok
    assert "25" in reason


def test_근거에_없는_날짜는_실패한다() -> None:
    ok, reason = rule_based_verify("2027년 3월 개정 기준으로 최대 1년입니다.", _CHUNKS)
    assert not ok
    assert "2027" in reason


def test_콤마_표기_차이는_허용된다() -> None:
    ok, _ = rule_based_verify("급여 상한은 월 1500000원입니다.", _CHUNKS)
    assert ok


def test_근거에_없는_문서명은_실패한다() -> None:
    ok, reason = rule_based_verify("자세한 내용은 취업규칙.pdf를 참고하세요.", _CHUNKS)
    assert not ok
    assert "취업규칙.pdf" in reason


def test_근거에_있는_문서명은_통과한다() -> None:
    ok, _ = rule_based_verify("휴가규정.pdf에 따르면 연차는 15일입니다.", _CHUNKS)
    assert ok


def test_빈_답변은_fail_closed() -> None:
    ok, _ = rule_based_verify("   ", _CHUNKS)
    assert not ok


def test_근거_청크가_없으면_fail_closed() -> None:
    ok, _ = rule_based_verify("연차는 15일입니다.", [])
    assert not ok

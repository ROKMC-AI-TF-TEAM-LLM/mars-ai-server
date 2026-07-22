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


# ── 목록 번호(서식) 오탐 방지 — 서식 숫자는 사실 주장이 아니다 ──────────────

_PROSE_CHUNKS = [
    # 숫자가 전혀 없는 산문 근거: 목록 번호 오탐이 가장 잘 드러나는 조건 (실측)
    {"text": "휴가에는 연차휴가와 병가가 있다.", "source_doc": "안내문"},
]


def test_목록_번호는_수치_검사에서_제외된다() -> None:
    ok, reason = rule_based_verify(
        "휴가 종류는 다음과 같습니다.\n1. 연차휴가\n2. 병가", _PROSE_CHUNKS
    )
    assert ok, reason


def test_헤더와_볼드_목록_번호도_제외된다() -> None:
    # 실측 답변 형식: "### 1." 헤더 번호, "**1. 항목**" 볼드 번호
    ok, reason = rule_based_verify(
        "## 개요\n### 1. 연차휴가\n내용 정리\n**2. 병가**\n설명", _PROSE_CHUNKS
    )
    assert ok, reason


def test_두_자리_목록_번호도_제외된다() -> None:
    items = "\n".join(f"{n}. 항목" for n in range(1, 13))  # 1. ~ 12.
    ok, reason = rule_based_verify(f"항목 목록:\n{items}", _PROSE_CHUNKS)
    assert ok, reason


def test_목록_항목_본문의_지어낸_수치는_여전히_실패한다() -> None:
    # 번호는 제외돼도 항목 내용 속 수치는 전수 검사 유지 (fail-closed)
    ok, reason = rule_based_verify("휴가 안내:\n1. 연차휴가는 매년 25일", _PROSE_CHUNKS)
    assert not ok
    assert "25" in reason


def test_줄_시작의_연도는_서식으로_오인하지_않는다() -> None:
    # 4자리 연도는 _LIST_NUMBERING_RE의 2자리 상한에 걸리지 않아 계속 검사된다
    ok, reason = rule_based_verify("2027. 1월 개정 기준입니다.", _CHUNKS)
    assert not ok
    assert "2027" in reason

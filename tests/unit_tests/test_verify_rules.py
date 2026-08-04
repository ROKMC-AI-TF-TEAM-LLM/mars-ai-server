"""rule_based_verify 유닛 테스트 — 숫자/날짜/문서명 검출 (roadmap 4단계 DoD)."""

from __future__ import annotations

from ax_rag.query_graph.nodes.verify import rule_based_verify
from ax_rag.shared.config import get_config

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


# ── 종합 허용 ① 유도 명시 — 답변이 계산 과정을 보여준 경우 ──────────────────
# 배경: 모든 수치가 근거에 문자열로 있어야 통과하던 규칙이 "총 20일" 같은 종합을
# 전부 탈락시켜 답변이 인용문 수준으로 짧아졌다 (정성 평가에서 13문항 중 7문항 재시도).


def test_유도_명시는_통과한다() -> None:
    """피연산자(15·5)가 근거에 있고 같은 문장에서 계산을 보여주면 통과."""
    ok, reason = rule_based_verify("연차 15일과 이월 5일을 합해 총 20일입니다.", _CHUNKS)
    assert ok, reason


def test_유도_피연산자가_근거에_없으면_탈락한다() -> None:
    """계산을 보여줘도 피연산자가 지어낸 값이면 fail-closed."""
    ok, reason = rule_based_verify("연차 30일과 이월 7일을 합해 총 37일입니다.", _CHUNKS)
    assert not ok


def test_유도가_다른_문장에_흩어져_있으면_조합_경로로만_판정된다() -> None:
    """①은 '같은 문장'을 요구한다 — 근거 제시로 볼 수 없기 때문."""
    chunks = [{"text": "가 항목은 40이다. 나 항목은 25이다.", "source_doc": "표"}]
    # 40·25가 같은 청크에 있으므로 ③으로는 통과한다 (65 = 40+25)
    ok, _ = rule_based_verify("가는 40입니다.\n나는 25입니다.\n합계는 65입니다.", chunks)
    assert ok


# ── 종합 허용 ② 열거 개수 ──────────────────────────────────────────────────


def test_열거_개수가_목록_항목_수와_일치하면_통과한다() -> None:
    ok, reason = rule_based_verify(
        "휴가 종류는 3가지입니다.\n- 연차휴가\n- 병가\n- 경조사휴가", _PROSE_CHUNKS
    )
    assert ok, reason


def test_열거_개수가_목록과_달라도_탈락시키지_않는다() -> None:
    """계수 오류는 '근거에 있는 사실인가'와 다른 문제다. 이걸로 탈락시켰더니
    정상 답변이 fallback으로 떨어지는 것을 실측했다 (LLM 검증이 잡는다)."""
    ok, reason = rule_based_verify("휴가 종류는 5가지입니다.\n- 연차휴가\n- 병가", _PROSE_CHUNKS)
    assert ok, reason


def test_개월은_열거_개수로_오인되지_않는다() -> None:
    """실측 사고: 초안의 '6개월씩 두 번 사용'에서 '6개'가 열거 개수로 잡혀
    정상 답변이 탈락했다. '개월'은 기간 단위이므로 근거 검사를 그대로 받는다."""
    ok, reason = rule_based_verify("6개월씩 나눠 쓸 수 있습니다.", _PROSE_CHUNKS)
    assert not ok
    assert "6" in reason


def test_개년_개소도_열거_개수로_오인되지_않는다() -> None:
    for text in ("3개년 계획입니다.", "12개소에서 운영합니다."):
        ok, reason = rule_based_verify(text, _PROSE_CHUNKS)
        assert not ok, f"{text} → {reason}"


def test_목록이_없는_개수_표현은_담화_표지로_제외된다() -> None:
    ok, reason = rule_based_verify("휴가 종류는 크게 2가지로 나뉩니다.", _PROSE_CHUNKS)
    assert ok, reason


def test_개수_단위가_아닌_규정_수치는_제외되지_않는다() -> None:
    """'일·년' 등 규정 값 단위는 ② 대상이 아니다 (지어낸 규정 수치 차단 유지)."""
    ok, reason = rule_based_verify("연차는 25일입니다.", _PROSE_CHUNKS)
    assert not ok
    assert "25" in reason


# ── 종합 허용 ③ 청크 내 조합 + 안전 포락선 ─────────────────────────────────

_ARITHMETIC_CHUNKS = [
    {"text": "가 수당은 40만원이고 나 수당은 25만원이다.", "source_doc": "수당표"},
    {"text": "다 수당은 11만원이다.", "source_doc": "수당표"},
]


def test_같은_청크_내_합은_통과한다() -> None:
    ok, reason = rule_based_verify("두 수당을 더하면 65만원입니다.", _ARITHMETIC_CHUNKS)
    assert ok, reason


def test_같은_청크_내_차도_통과한다() -> None:
    ok, reason = rule_based_verify("두 수당의 차이는 15만원입니다.", _ARITHMETIC_CHUNKS)
    assert ok, reason


def test_다른_청크의_수치_조합은_탈락한다() -> None:
    """안전 포락선: 조합 범위를 청크 안으로 가둔다 (40 + 11 = 51은 청크가 다르다)."""
    ok, reason = rule_based_verify("합계는 51만원입니다.", _ARITHMETIC_CHUNKS)
    assert not ok
    assert "51" in reason


def test_한_자리_수는_조합_경로에서_제외된다() -> None:
    """우연 일치가 흔해 제외한다 (40 - 25 = 15는 되지만 한 자리는 안 됨)."""
    chunks = [{"text": "가 항목 12, 나 항목 9이다.", "source_doc": "표"}]
    ok, reason = rule_based_verify("차이는 3입니다.", chunks)  # 12 - 9 = 3
    assert not ok
    assert "3" in reason


def test_수치가_너무_많은_청크는_조합_경로가_비활성된다() -> None:
    """조합 폭발 차단: 청크 내 수치가 상한(12개)을 넘으면 ③을 쓰지 않는다."""
    many = ", ".join(str(n) for n in range(100, 115))  # 15개
    chunks = [{"text": f"항목: {many}", "source_doc": "표"}]
    ok, reason = rule_based_verify("합계는 213입니다.", chunks)  # 100 + 113
    assert not ok
    assert "213" in reason


def test_스위치를_끄면_조합_경로가_막힌다(monkeypatch) -> None:
    """VERIFY_ALLOW_ARITHMETIC=false로 ③만 즉시 차단할 수 있다."""
    monkeypatch.setenv("VERIFY_ALLOW_ARITHMETIC", "false")
    get_config.cache_clear()
    try:
        ok, reason = rule_based_verify("두 수당을 더하면 65만원입니다.", _ARITHMETIC_CHUNKS)
        assert not ok
        assert "65" in reason
    finally:
        get_config.cache_clear()


def test_스위치를_꺼도_유도_명시와_열거_개수는_통과한다(monkeypatch) -> None:
    """①②는 답변 자체로 검증되는 안전 경로라 스위치와 무관하다."""
    monkeypatch.setenv("VERIFY_ALLOW_ARITHMETIC", "false")
    get_config.cache_clear()
    try:
        ok, reason = rule_based_verify("연차 15일과 이월 5일을 합해 총 20일입니다.", _CHUNKS)
        assert ok, reason
        ok, reason = rule_based_verify("종류는 2가지입니다.\n- 연차휴가\n- 병가", _PROSE_CHUNKS)
        assert ok, reason
    finally:
        get_config.cache_clear()


# ── 회귀 방지: 완화 후에도 지어낸 수치는 여전히 막혀야 한다 ─────────────────


def test_회귀_지어낸_규정_수치는_여전히_탈락한다() -> None:
    ok, reason = rule_based_verify("연차휴가는 매년 25일 부여됩니다.", _CHUNKS)
    assert not ok
    assert "25" in reason


def test_회귀_지어낸_연도는_여전히_탈락한다() -> None:
    ok, reason = rule_based_verify("2027년 개정 기준입니다.", _CHUNKS)
    assert not ok
    assert "2027" in reason


def test_회귀_연도는_유도로_통과할_수_없다() -> None:
    """실측 결함: 근거의 '2026년'과 '1년'으로 2026+1=2027이 성립해 지어낸
    개정 연도가 통과했다. 연도는 어떤 유도 경로로도 열지 않는다."""
    # ③ 조합 경로
    ok, reason = rule_based_verify("2027년 기준입니다.", _CHUNKS)
    assert not ok
    assert "2027" in reason
    # ① 유도 명시 경로 — 계산을 보여줘도 연도는 막힌다
    ok, reason = rule_based_verify("2026년에서 1년이 지나 2027년입니다.", _CHUNKS)
    assert not ok
    assert "2027" in reason


def test_회귀_작은_피연산자로_임의_값을_만들_수_없다() -> None:
    """근거에 1·2가 있어도 (X-1)+1 꼴로 임의 수치를 통과시키지 못한다.
    피연산자에도 하한(10)을 두어 ±1 미세 조정을 차단한다."""
    chunks = [{"text": "기준은 50이고 예외는 1건이다.", "source_doc": "표"}]
    ok, reason = rule_based_verify("한도는 51입니다.", chunks)  # 50 + 1
    assert not ok
    assert "51" in reason


def test_회귀_근거에_없는_문서명은_여전히_탈락한다() -> None:
    ok, reason = rule_based_verify("자세한 내용은 취업규칙.pdf를 참고하세요.", _CHUNKS)
    assert not ok
    assert "취업규칙.pdf" in reason

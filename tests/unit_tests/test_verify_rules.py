"""verify 전제 검사 유닛 테스트 — LLM에 판정을 물을 수 있는 상태인지만 본다.

수치·문서명이 근거에 문자열로 있는지 보던 규칙 검증은 제거했다. 부분 문자열
대조라 정밀도가 낮았고("150"이 "1500"에 매칭), 종합 표현을 살리려 예외 경로를
덧대는 과정에서 오탐·오통과가 반복됐다 (목록 번호 오탐, "6개월"을 개수로 오인,
근거의 2026+1로 지어낸 연도 2027이 통과).

근거 여부 판정은 LLM 검증이 전담한다 — 지어낸 일수/이월 한도/기한/조항 번호를
3회씩 전부 grounded=False로 잡는 것을 실측했고(15/15), 규칙이 못 하던
"근거는 15인데 25라고 했다"는 모순까지 판별한다.
"""

from __future__ import annotations

from ax_rag.query_graph.nodes.verify import check_preconditions

_CHUNKS = [
    {
        "text": "연차휴가는 매년 15일이 부여되며 최대 5일까지 이월할 수 있다.",
        "source_doc": "휴가규정.pdf",
    },
]


def test_답변과_근거가_있으면_검증_가능() -> None:
    ok, reason = check_preconditions("연차휴가는 매년 15일입니다.", _CHUNKS)
    assert ok, reason


def test_빈_답변은_fail_closed() -> None:
    ok, reason = check_preconditions("   ", _CHUNKS)
    assert not ok
    assert "비어" in reason


def test_근거_청크가_없으면_fail_closed() -> None:
    ok, reason = check_preconditions("연차는 15일입니다.", [])
    assert not ok
    assert "근거" in reason


def test_전제_검사는_내용의_근거_여부를_판단하지_않는다() -> None:
    """지어낸 수치라도 전제 검사는 통과시킨다 — 판정은 LLM 검증의 몫이다.

    여기서 막으려 들면 부분 문자열 대조로 되돌아가고, 그때 겪은 오탐
    (목록 번호·"6개월"·연도 조합)이 다시 생긴다.
    """
    ok, _ = check_preconditions("연차휴가는 매년 25일 부여됩니다.", _CHUNKS)
    assert ok

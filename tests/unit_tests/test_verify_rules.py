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

from ax_rag.query_graph.nodes.verify import check_preconditions, prune_unsupported

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


# ── 부분 수용: 근거 없는 문장만 덜어낸다 ──────────────────────────────────
# 실측: 병가 한도·진단서·급여는 정확한데 "지휘관 승인 필요"를 지어내
# 답변 545자가 통째로 fallback이 됐다. 지목된 문장만 빼고 나머지를 살린다.

_DRAFT = (
    "병가는 연 60일 한도입니다.\n"
    "진단서 제출이 필요하며 3일 이내는 생략할 수 있습니다.\n"
    "병가 신청은 일반적으로 지휘관의 승인을 받아야 합니다.\n"
    "급여는 취업규칙이 정하는 바에 따릅니다."
)


def test_지목된_문장만_제거된다() -> None:
    pruned = prune_unsupported(_DRAFT, ["병가 신청은 일반적으로 지휘관의 승인을 받아야 합니다."])
    assert "지휘관의 승인" not in pruned
    assert "연 60일 한도" in pruned  # 근거 있는 내용은 남는다
    assert "취업규칙" in pruned


def test_초안에_없는_지목은_건너뛴다() -> None:
    """검증기가 요약·변형해 지목하면 어림짐작으로 지우지 않는다."""
    assert prune_unsupported(_DRAFT, ["지휘관 승인 관련 내용"]) == _DRAFT


def test_너무_짧은_지목은_무시된다() -> None:
    """조각 매칭으로 본문이 깎이는 것을 막는다."""
    assert prune_unsupported(_DRAFT, ["병가"]) == _DRAFT


def test_너무_많이_잘리면_부분_수용을_포기한다() -> None:
    """답변 대부분이 근거 밖이면 재생성·fallback이 맞다."""
    result = prune_unsupported(_DRAFT, [_DRAFT[:-20]])
    assert result == _DRAFT


def test_지목이_없으면_원본_그대로() -> None:
    assert prune_unsupported(_DRAFT, []) == _DRAFT


def test_제거_후_빈_목록_기호가_남지_않는다() -> None:
    draft = "- 연 60일 한도입니다.\n- 지휘관 승인이 필요합니다.\n- 급여는 취업규칙에 따릅니다."
    pruned = prune_unsupported(draft, ["지휘관 승인이 필요합니다."])
    assert "지휘관" not in pruned
    assert not any(line.strip() in ("-", "*", "•") for line in pruned.splitlines())


def test_전제_검사는_내용의_근거_여부를_판단하지_않는다() -> None:
    """지어낸 수치라도 전제 검사는 통과시킨다 — 판정은 LLM 검증의 몫이다.

    여기서 막으려 들면 부분 문자열 대조로 되돌아가고, 그때 겪은 오탐
    (목록 번호·"6개월"·연도 조합)이 다시 생긴다.
    """
    ok, _ = check_preconditions("연차휴가는 매년 25일 부여됩니다.", _CHUNKS)
    assert ok

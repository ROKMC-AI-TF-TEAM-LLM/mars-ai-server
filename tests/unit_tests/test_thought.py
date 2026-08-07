"""thought 위생 처리 유닛 테스트.

thought는 **verify를 거치지 않고 사용자 화면에 표시되는 유일한 LLM 텍스트**다.
프롬프트 지시가 아니라 코드로 강제되는 부분(길이·개행·꺾쇠)을 여기서 고정한다.
"""

from __future__ import annotations

from ax_rag.query_graph.thought import MAX_THOUGHT_CHARS, sanitize_thought


def test_짧은_문장은_그대로_통과한다() -> None:
    assert sanitize_thought("휴가 규정을 문서에서 찾는다") == "휴가 규정을 문서에서 찾는다"


def test_길이_상한을_넘으면_자른다() -> None:
    """상한은 프롬프트 지시가 아니라 코드가 강제한다 — 긴 문장은 곧 내용 서술이다."""
    long_thought = "가" * 300
    result = sanitize_thought(long_thought)
    assert len(result) == MAX_THOUGHT_CHARS + 1  # 말줄임표 한 글자
    assert result.endswith("…")


def test_개행과_제어문자는_공백으로_바뀐다() -> None:
    """SSE 프레임은 JSON 한 줄이다 — 개행이 남으면 표시가 깨진다."""
    assert sanitize_thought("첫 줄\n둘째 줄\t셋째") == "첫 줄 둘째 줄 셋째"
    assert "\n" not in sanitize_thought("a\r\n\r\nb")


def test_꺾쇠는_제거된다() -> None:
    """관측 안의 delimiter가 thought로 새어 나오는 것을 막는다."""
    assert (
        sanitize_thought("<observation>내용</observation>을 봤다")
        == "observation내용/observation을 봤다"
    )


def test_빈_값은_빈_문자열이_된다() -> None:
    """호출부가 기본 문구로 대체하므로 스트림은 끊기지 않는다."""
    assert sanitize_thought("") == ""
    assert sanitize_thought(None) == ""
    assert sanitize_thought("   \n  ") == ""

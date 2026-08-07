"""검색어 위생 처리 유닛 테스트 (query_hygiene.py).

원래 라우터 안에 있던 규칙이며, 지금은 에이전트가 만든 검색어에 적용된다.
실측: 재작성 쿼리에 "문서로 만들어줘"가 남으면 리랭크 최고점이
0.738 → 0.022로 무너져 검색이 전멸한다.
"""

from __future__ import annotations

import pytest

from ax_rag.query_graph.query_hygiene import (
    MATCHER_ONLY_MAX_CHARS,
    has_search_hint,
    strip_file_phrases,
)


@pytest.mark.parametrize(
    ("polluted", "expected"),
    [
        ("해병대 관련 내용을 조사하여 문서로 만들어줘", "해병대 관련 내용"),
        ("휴가 규정 찾아서 파일로 만들어줘", "휴가 규정"),
        ("탄약 훈령 요약해서 한글 파일로 저장해줘", "탄약 훈령"),
        ("위 내용을 문서화해줘", "위 내용"),
        ("휴가 제도", "휴가 제도"),  # 파일 표현 없으면 그대로 (명사 "제도" 보존)
        ("문서 검색해줘", "문서 검색해줘"),  # "문서"만으로는 제거 안 함
    ],
)
def test_파일요청_표현을_걷어낸다(polluted: str, expected: str) -> None:
    assert strip_file_phrases(polluted) == expected


def test_전부_제거되면_원본_유지() -> None:
    """검색어가 실종될 정도로 깎이면 오염된 원본이라도 유지한다 (빈 쿼리 방지)."""
    assert strip_file_phrases("문서로 만들어줘") == "문서로 만들어줘"


@pytest.mark.parametrize(
    "question",
    [
        "규정 찾아서 한글 파일로 저장해줘",
        "해병대에 대해서 조사하고 한글 문서로 만들어줘",  # "조사" — 실측 유실 케이스
        "휴가 규정 정리해서 한글 문서로 저장해줘",
        "탄약 훈령 요약해서 한글 파일로 만들어줘",
    ],
)
def test_검색_선행_표현을_감지한다(question: str) -> None:
    """이 신호가 있으면 매처 단독으로 끝내지 않는다 — 검색 없이 저장만
    실행되던 사고(28자 질문)의 방지선."""
    assert has_search_hint(question) is True


def test_순수_도구_요청에는_검색_신호가_없다() -> None:
    assert has_search_hint("이 답변 한글 파일로 저장해줘") is False
    assert has_search_hint("전역까지 며칠 남았어?") is False


def test_매처_단독_종결_기준은_짧은_질문이다() -> None:
    assert len("전역까지 며칠 남았어?") <= MATCHER_ONLY_MAX_CHARS

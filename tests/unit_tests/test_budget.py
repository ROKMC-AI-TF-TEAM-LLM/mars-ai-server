"""query_graph/budget.py 유닛 테스트 — 이력 상한 준수 (roadmap 4단계 DoD)."""

from __future__ import annotations

from ax_rag.query_graph.budget import approx_tokens, trim_history


def _turn(role: str, chars: int) -> dict:
    return {"role": role, "content": "가" * chars}


def test_상한_내_이력은_그대로_유지된다() -> None:
    history = [_turn("user", 100), _turn("assistant", 100)]
    assert trim_history(history, max_tokens=1500) == history


def test_상한_초과분은_오래된_턴부터_잘린다() -> None:
    # 각 턴 220자 ≈ 100토큰, 상한 250토큰 → 최근 2턴만 남는다
    history = [
        {"role": "user", "content": "첫" * 220},
        {"role": "assistant", "content": "둘" * 220},
        {"role": "user", "content": "셋" * 220},
    ]
    trimmed = trim_history(history, max_tokens=250)
    assert len(trimmed) == 2
    assert trimmed[0]["content"].startswith("둘")  # 최근 2개, 순서 보존
    assert trimmed[1]["content"].startswith("셋")


def test_총_토큰이_상한을_넘지_않는다() -> None:
    history = [_turn("user", 300) for _ in range(20)]
    trimmed = trim_history(history, max_tokens=1500)
    total = sum(approx_tokens(m["content"]) for m in trimmed)
    assert total <= 1500
    assert 0 < len(trimmed) < 20


def test_최근_턴_하나가_상한을_넘으면_빈_리스트() -> None:
    history = [_turn("user", 10), _turn("assistant", 5000)]
    assert trim_history(history, max_tokens=100) == []


def test_빈_이력과_None_안전() -> None:
    assert trim_history([], max_tokens=1500) == []
    assert trim_history(None, max_tokens=1500) == []  # type: ignore[arg-type]

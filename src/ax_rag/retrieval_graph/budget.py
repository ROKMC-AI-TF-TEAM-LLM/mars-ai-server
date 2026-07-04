"""컨텍스트 토큰 예산 계산 + 대화 이력 절삭 (architecture.md §7).

대화 이력 상한 1,500토큰. 부모 크기 x top_n이 지배 변수이므로 청킹
파라미터를 바꾸면 architecture.md의 토큰 예산 표를 재계산할 것.
"""

from __future__ import annotations

import math

from ax_rag.shared.config import CHARS_PER_TOKEN


def approx_tokens(text: str) -> int:
    """한국어 토큰 수 근사: 문자수 / 2.2 올림."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def trim_history(history: list[dict], max_tokens: int = 1500) -> list[dict]:
    """최근 턴부터 역순으로 채우고 상한 초과분 절삭. 문자수/2.2 근사 사용.

    턴 단위로 자르며, 가장 최근 턴 하나만으로 상한을 넘으면 빈 리스트가 된다.
    """
    kept_reversed: list[dict] = []
    total = 0
    for message in reversed(history or []):
        cost = approx_tokens(message.get("content", ""))
        if total + cost > max_tokens:
            break
        kept_reversed.append(message)
        total += cost
    return list(reversed(kept_reversed))

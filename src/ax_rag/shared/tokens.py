"""한국어 토큰 수 근사 (두 그래프 공용).

컨텍스트 예산 계산(query_graph/budget.py)과 청킹 크기 산정
(indexer_graph/chunking.py)이 같은 근사식을 써야 architecture.md의
토큰 예산 표와 실제 동작이 어긋나지 않는다. 그래서 shared에 둔다.

근사 계수는 config.CHARS_PER_TOKEN (L40에서 실측 보정 예정).
"""

from __future__ import annotations

import math

from ax_rag.shared.config import CHARS_PER_TOKEN


def approx_tokens(text: str) -> int:
    """한국어 토큰 수 근사: 문자수 / 2.2 올림."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)

"""SSE 프레임 조립과 답변 분할 전송 (interfaces.md §4·§5).

토큰 실시간 스트리밍이 아니라 **verify 통과 후 분할 전송**이다
(architecture.md §8): 검증 실패 시 재생성되므로, 이미 보낸 텍스트를
취소할 수 없는 SSE 계약과 맞추려면 확정 후에 쪼개 보내야 한다.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# text 이벤트 분할: 문장 경계가 없을 때의 조각 길이 상한 (interfaces.md §4)
_MAX_PIECE_CHARS = 80

# 문장 경계 분할: 마침표류 뒤 공백까지 포함해 세그먼트를 만들고,
# 이어 붙이면 원문과 동일해지도록 잔여 텍스트도 세그먼트로 잡는다
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+\s*|[^.!?]+\s*$")


def sse_event(payload: dict) -> str:
    """dict → 'data: {json}\n\n' SSE 프레임. ensure_ascii=False."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def split_for_stream(text: str) -> list[str]:
    """문장 경계 우선(다./요./.), 경계가 없으면 80자 내외로 분할.

    조각을 전부 이어 붙이면 원문과 동일하다 (공백 보존).
    """
    pieces: list[str] = []
    for match in _SENTENCE_RE.finditer(text):
        segment = match.group(0)
        if not segment:
            continue
        while len(segment) > _MAX_PIECE_CHARS:
            pieces.append(segment[:_MAX_PIECE_CHARS])
            segment = segment[_MAX_PIECE_CHARS:]
        if segment:
            pieces.append(segment)
    return pieces


async def stream_answer(
    final_answer: str,
    sources: list[dict],
    files: list[dict] | None = None,
    notice: dict | None = None,
) -> AsyncIterator[str]:
    """확정된 답변을 text로 분할 전송 → file 0회 이상 → notice 0~1회 → sources 1회 → done.

    file은 도구가 만든 파일의 구조화 신호다 — 미들웨어가 답변 텍스트를 파싱하지
    않고 이 이벤트로 fetch-and-store 한다. notice는 답변 자체에 대한 경고이며,
    텍스트에 섞지 않고 별도 타입으로 보내 프론트가 다르게 표시하게 한다
    (sources가 done 직전 마지막이어야 하므로 그 앞에 온다).
    """
    interval_seconds = get_config().STREAM_TEXT_INTERVAL_MS / 1000
    pieces = split_for_stream(final_answer)
    logger.debug("SSE 스트리밍 시작: text %d조각, sources %d건", len(pieces), len(sources))
    for index, piece in enumerate(pieces, start=1):
        logger.debug("SSE text [%d/%d] (%d자): %s", index, len(pieces), len(piece), piece)
        yield sse_event({"type": "text", "content": piece})
        await asyncio.sleep(interval_seconds)
    for file_info in files or []:
        logger.debug("SSE file: %s", file_info.get("name"))
        yield sse_event({"type": "file", **file_info})
    if notice:
        logger.debug("SSE notice: %s", notice.get("code"))
        yield sse_event({"type": "notice", **notice})
    logger.debug("SSE sources: %s", [s["name"] for s in sources])
    yield sse_event({"type": "sources", "items": sources})
    logger.debug("SSE done")
    yield sse_event({"type": "done"})

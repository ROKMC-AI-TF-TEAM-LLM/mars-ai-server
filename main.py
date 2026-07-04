"""사내 인트라넷 연동용 FastAPI 래퍼 (포트 9000, interfaces.md §5).

미들웨어와의 계약:
- 요청은 JSON(question, user_department, messages[human|ai])
- 응답은 SSE(text/event-stream): text 이벤트 N회 → sources 1회 → data: [DONE]
- 토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송이다 (architecture.md §8)

★ 단일 워커 강제: Milvus Lite 파일 락 충돌 때문에 uvicorn 멀티 워커 금지.
  실행: uvicorn main:app --host 0.0.0.0 --port 9000   (--workers 옵션 사용 금지)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ax_rag.retrieval_graph.graph import graph
from ax_rag.shared.audit_log import log_query
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger, setup_logging

# uvicorn으로 실행돼도 통일 포맷(시각 | 레벨 | 모듈 | 메시지)이 적용되게 임포트 시점에 설정
setup_logging()
logger = get_logger("main")

app = FastAPI(title="A.X RAG 서버")

# 브라우저 프론트가 직접 붙는 개발/데모용 CORS 허용.
# 운영 경로는 미들웨어의 서버 간 호출이라 CORS가 관여하지 않는다 (내부망 신뢰)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# text 이벤트 분할: 문장 경계가 없을 때의 조각 길이 상한 (interfaces.md §4)
_MAX_PIECE_CHARS = 80

# 문장 경계 분할: 마침표류 뒤 공백까지 포함해 세그먼트를 만들고,
# 이어 붙이면 원문과 동일해지도록 잔여 텍스트도 세그먼트로 잡는다
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+\s*|[^.!?]+\s*$")


class QueryRequest(BaseModel):
    """POST /query 요청 (미들웨어 계약)."""

    question: str
    # 누락 시 가장 제한적으로 처리: visibility ALL 문서만 검색 (interfaces.md §5)
    user_department: str = ""
    messages: list[dict] = Field(default_factory=list)


def to_internal_history(messages: list[dict]) -> list[dict]:
    """미들웨어 role("human"|"ai") → 내부 role("user"|"assistant") 변환.
    알 수 없는 role은 건너뛰고 warning 로그."""
    converted: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "human":
            converted.append({"role": "user", "content": message.get("content", "")})
        elif role == "ai":
            converted.append({"role": "assistant", "content": message.get("content", "")})
        else:
            logger.warning("알 수 없는 role을 건너뛴다: %r", role)
    return converted


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


async def stream_answer(final_answer: str, sources: list[dict]) -> AsyncIterator[str]:
    """확정된 답변을 text 이벤트로 분할 전송 → sources 1회 → 'data: [DONE]\n\n'.

    조각 사이 간격(config.STREAM_TEXT_INTERVAL_MS)을 두어 TCP 병합 없이
    순차 도착하게 하고, 프론트에서 타자기 효과가 보이게 한다.
    """
    interval_seconds = get_config().STREAM_TEXT_INTERVAL_MS / 1000
    pieces = split_for_stream(final_answer)
    logger.debug("SSE 스트리밍 시작: text %d조각, sources %d건", len(pieces), len(sources))
    for index, piece in enumerate(pieces, start=1):
        logger.debug("SSE text [%d/%d] (%d자): %s", index, len(pieces), len(piece), piece)
        yield sse_event({"type": "text", "content": piece})
        await asyncio.sleep(interval_seconds)
    logger.debug("SSE sources: %s", [s["name"] for s in sources])
    yield sse_event({"type": "sources", "items": sources})
    logger.debug("SSE [DONE]")
    yield "data: [DONE]\n\n"


def _build_sources(retrieved_chunks: list[dict], grounded: bool) -> list[dict]:
    """근거 청크에서 중복 없는 sources 목록을 만든다.

    sources는 "답변의 근거로 실제 사용된 문서"다. verify를 통과하지 못한
    답변(fallback)은 검색 결과를 근거로 쓰지 않았으므로 빈 목록을 반환한다
    (예: "안녕" 같은 잡담에 검색 상위 문서가 출처로 노출되는 것 방지).
    page는 청크 메타데이터에 페이지 정보가 없으므로 null (미확정 항목).
    """
    if not grounded:
        return []
    sources: list[dict] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        name = chunk.get("source_doc")
        if name and name not in seen:
            seen.add(name)
            sources.append({"name": name, "page": None})
    return sources


async def _run_pipeline(request: QueryRequest) -> AsyncIterator[str]:
    """그래프를 완주(invoke)한 뒤 확정 답변을 SSE로 분할 전송한다.

    fallback 답변도 정상 text로 보낸다. error 이벤트는 파이프라인
    예외(서비스 다운, 타임아웃 등)에만 사용하고, 스트림은 항상 [DONE]으로 끝난다.
    """
    user_department = request.user_department or ""
    try:
        state = {
            "question": request.question,
            "user_department": user_department,
            "conversation_history": to_internal_history(request.messages),
        }
        # 동기 그래프를 스레드로 넘겨 이벤트 루프를 막지 않는다
        result = await asyncio.to_thread(graph.invoke, state)

        grounded = bool(result.get("grounded"))
        sources = _build_sources(result.get("retrieved_chunks") or [], grounded)
        log_query(
            user_department=user_department,
            question=request.question,
            domain=result.get("domain") or "GENERAL",
            sources=[s["name"] for s in sources],
            grounded=grounded,
        )
        async for frame in stream_answer(result.get("final_answer") or "", sources):
            yield frame
    except Exception:
        logger.exception("파이프라인 예외")
        try:
            log_query(
                user_department=user_department,
                question=request.question,
                domain="GENERAL",
                sources=[],
                grounded=False,
            )
        except Exception:
            logger.exception("감사 로그 기록 실패")
        yield sse_event({"type": "error", "message": "내부 오류로 답변을 생성하지 못했습니다."})
        yield "data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    """헬스체크."""
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """질의응답 SSE 엔드포인트 (미들웨어 전용)."""
    return StreamingResponse(
        _run_pipeline(request),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # 리버스 프록시 버퍼링 방지 (필수)
            "Cache-Control": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # 단일 워커 강제 (Milvus Lite 파일 락). workers 인자를 절대 늘리지 말 것
    uvicorn.run(app, host="0.0.0.0", port=9000)

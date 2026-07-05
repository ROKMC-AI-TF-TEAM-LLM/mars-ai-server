"""사내 인트라넷 연동용 FastAPI 래퍼 (포트 9000, interfaces.md §5).

미들웨어와의 계약:
- 요청은 JSON(question, user_department, messages[human|ai])
- 응답은 SSE(text/event-stream): text 이벤트 N회 → sources 1회 → {"type": "done"}
- 토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송이다 (architecture.md §8)

★ 단일 워커 강제: Milvus Lite 파일 락 충돌 때문에 uvicorn 멀티 워커 금지.
  실행: uvicorn main:app --host 0.0.0.0 --port 9000   (--workers 옵션 사용 금지)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ax_rag.retrieval_graph.graph import graph
from ax_rag.shared.audit_log import log_query
from ax_rag.shared.config import DOMAINS, SMALLTALK_DOMAIN, get_config
from ax_rag.shared.logging_setup import get_logger, setup_logging

# uvicorn으로 실행돼도 통일 포맷(시각 | 레벨 | 모듈 | 메시지)이 적용되게 임포트 시점에 설정
setup_logging()
logger = get_logger("main")

app = FastAPI(
    title="A.X RAG 서버",
    version="0.1.0",
    description=(
        "사내 업무 문서 검색 챗봇 API (미들웨어 연동용).\n\n"
        "- 응답은 **SSE(text/event-stream) 스트리밍** — 이벤트 계약은 `POST /query` 문서 참조\n"
        "- 로컬 서비스 4종(vLLM :8000, 임베딩 :8001, 리랭커 :8002, Milvus)이 기동되어 있어야 한다\n"
        "- 상세 스펙: docs/interfaces.md §5"
    ),
)

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
    """POST /query 요청 본문 (미들웨어 계약, interfaces.md §5)."""

    question: str = Field(
        description="사용자 질문 (자연어, 구어체 허용)",
        examples=["육아휴직은 얼마나 쓸 수 있어?"],
    )
    user_department: str = Field(
        default="",
        description=(
            "ACL 판정 근거 부서 코드. 누락 시 가장 제한적으로 처리되어 "
            "visibility=ALL 문서만 검색된다 (DEPT_ONLY 전부 배제)"
        ),
        examples=["HR_TEAM"],
    )
    domain: str = Field(
        default="",
        description=(
            "검색 도메인 한정 (선택). 허용값: HR | TECH | FINANCE_LEGAL. "
            '빈 값 또는 "ALL"이면 도메인 무관 검색. "GENERAL"·미지의 값도 '
            "도메인 무관으로 처리. 라우터의 LLM 분류는 검색 범위를 제한하지 않는다"
        ),
        examples=["FINANCE_LEGAL"],
    )
    messages: list[dict] = Field(
        default_factory=list,
        description='이전 대화 이력. role은 미들웨어 규약 "human" | "ai"',
        examples=[
            [
                {"role": "human", "content": "육아휴직에 대해 알려줘"},
                {"role": "ai", "content": "육아휴직은 최대 1년까지 사용할 수 있습니다."},
            ]
        ],
    )


def normalize_requested_domain(value: str) -> str:
    """요청 domain 정규화: DOMAINS의 실제 도메인 값만 한정 필터로 인정한다.

    빈 값 / "ALL" / "GENERAL" / 미지의 값 → ""(도메인 무관 검색).
    """
    normalized = (value or "").strip().upper()
    if not normalized or normalized in ("ALL", "GENERAL"):
        return ""
    if normalized in DOMAINS:
        return normalized
    logger.warning("미지의 domain 요청값 무시: %r → 도메인 무관 검색", value)
    return ""


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
    """확정된 답변을 text 이벤트로 분할 전송 → sources 1회 → done 이벤트로 종료.

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
    logger.debug("SSE done")
    yield sse_event({"type": "done"})


def _status_after_node(node_name: str, merged_state: dict) -> tuple[str, str] | None:
    """그래프 노드 완료 시 다음 단계 안내 (stage, message). 안내가 없는 노드는 None.

    프론트가 "검색하는 중..." 같은 진행 상태를 표시할 수 있게 하는
    status 이벤트의 재료다. 완료된 노드를 보고 "이제 시작되는 단계"를 알린다.
    """
    if node_name == "route":
        if merged_state.get("domain") == SMALLTALK_DOMAIN:
            return ("generate", "답변을 생성하는 중...")
        return ("retrieve", "사내 문서를 검색하는 중...")
    if node_name == "fuse":
        return ("rerank", "관련 문서를 선별하는 중...")
    if node_name == "rerank":
        return ("generate", "답변을 생성하는 중...")
    if node_name == "generate":
        return ("verify", "답변이 문서에 근거하는지 검증하는 중...")
    if node_name == "increment_retry":
        return ("generate", "답변을 다시 생성하는 중...")
    return None


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


async def _run_pipeline(request: QueryRequest, http_request: Request) -> AsyncIterator[str]:
    """그래프를 완주(invoke)한 뒤 확정 답변을 SSE로 분할 전송한다.

    fallback 답변도 정상 text로 보낸다. error 이벤트는 파이프라인
    예외(서비스 다운, 타임아웃 등)에만 사용하고, 스트림은 항상 done 이벤트로 끝난다.

    생성 중지: 별도 API 없이 클라이언트의 SSE 연결 중단으로 처리한다.
    노드 경계마다 연결을 확인해 끊겼으면 이후 단계를 실행하지 않는다
    (진행 중이던 단일 LLM 호출까지는 완료됨 — 강제 중단 불가).
    """
    user_department = request.user_department or ""
    try:
        state = {
            "question": request.question,
            "user_department": user_department,
            "requested_domain": normalize_requested_domain(request.domain),
            "conversation_history": to_internal_history(request.messages),
        }

        # 그래프를 노드 단위로 진행시키며(stream) 단계마다 status 이벤트를 흘린다.
        # 동기 제너레이터의 next()만 스레드로 넘겨 이벤트 루프를 막지 않는다
        yield sse_event({"type": "status", "stage": "route", "message": "질문을 분석하는 중..."})
        updates_stream = graph.stream(state, stream_mode="updates")
        sentinel = object()
        result: dict = dict(state)
        while True:
            update = await asyncio.to_thread(next, updates_stream, sentinel)
            if update is sentinel:
                break
            if await http_request.is_disconnected():
                logger.info(
                    "클라이언트 연결 중단 감지 → 파이프라인 중단 (질문=%s)", request.question
                )
                updates_stream.close()  # 이후 노드 실행 방지
                return
            for node_name, delta in update.items():
                if isinstance(delta, dict):
                    result.update(delta)  # 노드별 변경분을 병합해 최종 상태를 복원
                status = _status_after_node(node_name, result)
                if status is not None:
                    stage, message = status
                    logger.debug("SSE status: %s (%s 완료 후)", message, node_name)
                    yield sse_event({"type": "status", "stage": stage, "message": message})

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
    except asyncio.CancelledError:
        # 클라이언트가 연결을 중단(abort)하면 Starlette가 이 제너레이터를 취소한다.
        # 진행 중이던 노드까지만 실행되고 이후 단계는 실행되지 않는다
        logger.info("클라이언트 연결 중단 → 파이프라인 중단 (질문=%s)", request.question)
        raise
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
        yield sse_event({"type": "done"})


@app.get("/health", summary="헬스체크")
def health() -> dict[str, str]:
    """서버 생존 확인. 파이프라인/모델 상태는 검사하지 않는다."""
    return {"status": "ok"}


_QUERY_RESPONSES = {
    200: {
        "description": (
            "SSE 스트림 (`data: {JSON}\\n\\n` 프레임). 이벤트 순서:\n\n"
            '1. `{"type": "status", "stage": str, "message": str}` — 진행 상태 안내, '
            '0회 이상 (예: "사내 문서를 검색하는 중..."). stage 값: '
            "route | retrieve | rerank | generate | verify\n"
            '2. `{"type": "text", "content": str}` — 답변 조각, N회 '
            "(문장 단위, STREAM_TEXT_INTERVAL_MS 간격)\n"
            '3. `{"type": "sources", "items": [{"name": str, "page": null}]}` — '
            "정확히 1회. 근거 검증(verify)을 통과한 답변에만 문서가 담기며, "
            "fallback·잡담 응답은 빈 배열\n"
            '4. `{"type": "done"}` — 종료 신호, 항상 마지막\n\n'
            '파이프라인 예외 시: `{"type": "error", "message": str}` → done. '
            "fallback 답변은 error가 아니라 정상 text로 전송된다. "
            "클라이언트는 미지의 type을 무시하도록 구현할 것 (향후 확장 대비)."
        ),
        "content": {
            "text/event-stream": {
                "example": (
                    'data: {"type": "text", "content": '
                    '"육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있습니다. "}\n\n'
                    'data: {"type": "sources", "items": '
                    '[{"name": "휴가규정.md", "page": null}]}\n\n'
                    'data: {"type": "done"}\n\n'
                )
            }
        },
    }
}


@app.post("/query", summary="질의응답 (SSE 스트리밍)", responses=_QUERY_RESPONSES)
async def query(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """질의응답 파이프라인을 완주한 뒤 확정 답변을 SSE로 분할 전송한다.

    처리 흐름: 라우팅(잡담 감지·쿼리 재작성) → 하이브리드 검색(벡터+BM25,
    부서 ACL 강제) → 리랭크 → 답변 생성 → 근거 검증(fail-closed) → 스트리밍.

    - 토큰 실시간 스트리밍이 아니라 **검증 통과 후 분할 전송**이다
      (검증 실패 시 재생성되므로, 이미 보낸 텍스트를 취소할 수 없는 SSE 계약과의 정합)
    - `domain`을 지정하면 해당 도메인 문서만 검색한다. 빈 값/"ALL"이면 전 도메인
    - `user_department` 누락 시 visibility=ALL 문서만 검색된다
    - **생성 중지**: 별도 API 없음. 클라이언트가 SSE 연결을 중단(abort)하면
      서버가 노드 경계에서 감지해 이후 단계를 실행하지 않는다.
      미들웨어는 프론트 연결 중단 시 본 서버로의 요청도 함께 중단해야 한다
    """
    logger.info(
        "질의 수신: dept=%s, domain=%s, 이력 %d턴, 질문=%s",
        request.user_department or "(없음)",
        request.domain or "(전체)",
        len(request.messages),
        request.question,
    )
    # 미들웨어 연동 디버깅용 요청 전문 (LOG_LEVEL=DEBUG일 때만 출력)
    logger.debug("요청 전문: %s", json.dumps(request.model_dump(), ensure_ascii=False))
    return StreamingResponse(
        _run_pipeline(request, http_request),
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

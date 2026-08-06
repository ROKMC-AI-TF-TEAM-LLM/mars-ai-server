"""질의 파이프라인 실행과 SSE 변환 (POST /query의 본체).

그래프를 노드 단위로 진행시키며 단계마다 status 이벤트를 흘리고, 완주 후
확정 답변을 분할 전송한다. 단계 안내 문구의 소유는 그래프 쪽이다
(query_graph/stages.py) — API가 노드 이름을 문자열로 알지 않게 한다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import Request

from ax_rag.api.normalize import normalize_requested_domain, normalize_tool, to_internal_history
from ax_rag.api.schemas import QueryRequest
from ax_rag.api.sse import sse_event, stream_answer
from ax_rag.query_graph.graph import graph
from ax_rag.query_graph.prompts import KNOWLEDGE_ANSWER_NOTICE
from ax_rag.query_graph.stages import status_after_node
from ax_rag.shared.audit_log import log_query
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


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


def _answer_notice(answer_mode: str | None) -> dict | None:
    """답변 경로에 따라 붙일 SSE notice 페이로드. 붙일 것이 없으면 None.

    지식 기반 답변(answer_mode="knowledge")은 검증을 거치지 않은 LLM 자체
    지식이므로 문서 근거가 없음을 사용자에게 알린다. 경고를 답변 텍스트에
    섞지 않고 별도 이벤트로 보내 프론트가 다르게 표시하게 한다.
    """
    if answer_mode == "knowledge":
        return dict(KNOWLEDGE_ANSWER_NOTICE)
    return None


async def run_pipeline(request: QueryRequest, http_request: Request) -> AsyncIterator[str]:
    """그래프를 완주(invoke)한 뒤 확정 답변을 SSE로 분할 전송한다.

    fallback 답변도 정상 text로 보낸다. error 이벤트는 파이프라인
    예외(서비스 다운, 타임아웃 등)에만 사용하고, 스트림은 항상 done 이벤트로 끝난다.

    생성 중지: 별도 API 없이 클라이언트의 SSE 연결 중단으로 처리한다.
    노드 경계마다 연결을 확인해 끊겼으면 이후 단계를 실행하지 않는다
    (진행 중이던 단일 LLM 호출까지는 완료됨 — 강제 중단 불가).
    """
    user_department = request.user_department or ""
    requested_domain = normalize_requested_domain(request.domain)
    try:
        state = {
            "question": request.question,
            "user_department": user_department,
            "requested_domain": requested_domain,
            "conversation_history": to_internal_history(request.messages),
        }
        forced_intent = normalize_tool(request.tool)
        if forced_intent:
            # 강제 경로: 라우터가 분류를 건너뛰고 이 값을 그대로 쓴다 (엄격 모드)
            state["intent"] = forced_intent

        # 그래프를 노드 단위로 진행시키며(stream) 단계마다 status 이벤트를 흘린다.
        # 동기 제너레이터의 next()만 스레드로 넘겨 이벤트 루프를 막지 않는다
        yield sse_event(
            {
                "type": "status",
                "stage": "route",
                "message": "질문을 분석하고 처리 계획을 수립하는 중...",
            }
        )
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
                status = status_after_node(node_name, result)
                if status is not None:
                    stage, message = status
                    logger.debug("SSE status: %s (%s 완료 후)", message, node_name)
                    yield sse_event({"type": "status", "stage": stage, "message": message})

        grounded = bool(result.get("grounded"))
        sources = _build_sources(result.get("retrieved_chunks") or [], grounded)
        answer_mode = result.get("answer_mode")
        log_query(
            user_department=user_department,
            question=request.question,
            # LLM 추측이 아니라 실제 적용된 검색 범위를 기록한다
            domain=requested_domain or "ALL",
            sources=[s["name"] for s in sources],
            grounded=grounded,
            answer_mode=answer_mode,
        )
        async for frame in stream_answer(
            result.get("final_answer") or "",
            sources,
            result.get("generated_files") or [],
            _answer_notice(answer_mode),
        ):
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
                domain=requested_domain or "ALL",
                sources=[],
                grounded=False,
            )
        except Exception:
            logger.exception("감사 로그 기록 실패")
        yield sse_event({"type": "error", "message": "내부 오류로 답변을 생성하지 못했습니다."})
        yield sse_event({"type": "done"})

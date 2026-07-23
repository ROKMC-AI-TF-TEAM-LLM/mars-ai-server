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
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ax_rag.indexer_graph import ingest
from ax_rag.indexer_graph.loaders import SUPPORTED_SUFFIXES
from ax_rag.query_graph.graph import graph
from ax_rag.query_graph.tools import (
    DEFAULT_TOOL_STATUS_MESSAGE,
    DOC_SEARCH,
    FORCIBLE_TOOLS,
    TERMINAL_ONLY_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_NODES,
    TOOL_STATUS_MESSAGES,
    execution_queue,
)
from ax_rag.shared import vectorstore
from ax_rag.shared.audit_log import log_query
from ax_rag.shared.config import DOMAIN_LABELS, DOMAINS, get_config
from ax_rag.shared.health import check_dependencies
from ax_rag.shared.ingest_jobs import IngestJobRegistry
from ax_rag.shared.logging_setup import get_logger, setup_logging

# uvicorn으로 실행돼도 통일 포맷(시각 | 레벨 | 모듈 | 메시지)이 적용되게 임포트 시점에 설정
setup_logging()
logger = get_logger("main")

# Swagger UI 그룹핑용 태그 (목적별 분류). 각 엔드포인트의 tags=와 이름을 맞춘다
_OPENAPI_TAGS = [
    {
        "name": "질의응답",
        "description": "사용자 질문 처리(SSE 스트리밍)와 요청에 넣을 수 있는 "
        "domain·tool 값 조회. 미들웨어의 채팅 경로.",
    },
    {
        "name": "문서 관리",
        "description": "RAG 문서 적재·갱신·목록·삭제와 적재 작업 상태. 관리자 "
        "페이지 데이터 소스(미들웨어가 관리자 권한 확인 후 프록시). "
        "상세 연동은 docs/middleware_document_ingest.md 참조.",
    },
    {
        "name": "생성 파일",
        "description": "도구(HWP_EXPORT 등)가 만든 문서 파일 다운로드. 미들웨어가 "
        "SSE file 이벤트를 신호로 가져가 자체 저장(fetch-and-store).",
    },
    {
        "name": "운영",
        "description": "서버·의존 서비스 상태 점검 (모니터링용).",
    },
]

app = FastAPI(
    title="A.X RAG 서버",
    version="0.1.0",
    description=(
        "군 내부 문서 검색 챗봇 MARS API (미들웨어 연동용).\n\n"
        "- 응답은 **SSE(text/event-stream) 스트리밍** — 이벤트 계약은 `POST /query` 문서 참조\n"
        "- 로컬 서비스 4종(vLLM :8000, 임베딩 :8001, 리랭커 :8002, Milvus)이 기동되어 있어야 한다\n"
        "- 상세 스펙: docs/interfaces.md §5"
    ),
    openapi_tags=_OPENAPI_TAGS,
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
            "검색 도메인 한정 (선택). 허용값: HR | TECH | FINANCE_LEGAL | "
            "MANUAL(교범) | DIRECTIVE(훈령). "
            '빈 값 또는 "ALL"이면 도메인 무관 검색. "GENERAL"·미지의 값도 '
            "도메인 무관으로 처리. 교범/훈령 한정 검색 모드는 이 필드로 구현한다"
        ),
        examples=["DIRECTIVE"],
    )
    tool: str = Field(
        default="",
        description=(
            "처리 경로 강제 지정 (선택). 허용값은 GET /capabilities의 forcible=true 항목 "
            "(현재: DOC_SEARCH, DISCHARGE_DAYS, HWP_EXPORT). "
            "지정하면 라우터의 자동 분류를 무시하고 해당 경로로 직행한다 (잡담 예외 없음). "
            "빈 값이면 자동 분류, 미지의 값·강제 비허용 도구(SMALLTALK 등)는 무시하고 자동 분류"
        ),
        examples=[""],
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


def normalize_tool(value: str) -> str:
    """요청 tool 정규화: DOC_SEARCH 또는 강제 허용(FORCIBLE) 도구만 인정한다.

    빈 값 / 미지의 값 / 강제 비허용 도구 → ""(라우터 자동 분류).
    SMALLTALK 등 강제 비허용 도구를 지정하면 무시된다 — 강제 잡담 경로로
    업무 질문이 들어오면 verify 없이 모델이 규정을 지어낼 수 있어서다 (실측).
    """
    normalized = (value or "").strip().upper()
    if not normalized:
        return ""
    if normalized == DOC_SEARCH or normalized in FORCIBLE_TOOLS:
        return normalized
    if normalized in TOOL_NODES:
        logger.warning("강제 지정이 허용되지 않는 도구: %r → 자동 분류", value)
        return ""
    logger.warning("미지의 tool 요청값 무시: %r → 자동 분류", value)
    return ""


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


async def stream_answer(
    final_answer: str, sources: list[dict], files: list[dict] | None = None
) -> AsyncIterator[str]:
    """확정된 답변을 text 이벤트로 분할 전송 → file 0회 이상 → sources 1회 → done.

    file 이벤트는 도구가 생성한 파일(HWPX 등)의 구조화 신호다 — 미들웨어가
    답변 텍스트를 정규식으로 파싱하지 않고 이 이벤트로 fetch-and-store 한다.
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
    for file_info in files or []:
        logger.debug("SSE file: %s", file_info.get("name"))
        yield sse_event({"type": "file", **file_info})
    logger.debug("SSE sources: %s", [s["name"] for s in sources])
    yield sse_event({"type": "sources", "items": sources})
    logger.debug("SSE done")
    yield sse_event({"type": "done"})


def _next_stage_status(merged_state: dict) -> tuple[str, str] | None:
    """실행 큐(pending_intents)의 선두를 보고 다음 단계 안내를 만든다.

    - 다음이 도구 → stage="tool" + 도구별 문구 (TOOL_STATUS_MESSAGES, 레지스트리 기반)
    - 다음이 DOC_SEARCH → stage="retrieve" (문서 검색 시작)
    - 큐 소진 → None (finalize는 즉시 끝나므로 안내 불필요)
    """
    pending = merged_state.get("pending_intents")
    if pending is None:  # 구형 상태: 계획 또는 대표 intent에서 재구성 (방어적)
        plan = merged_state.get("intents")
        if plan:
            pending = execution_queue(list(plan))
        else:
            intent = merged_state.get("intent")
            pending = [intent] if intent else [DOC_SEARCH]
    if not pending:
        return None
    head = pending[0]
    if head == DOC_SEARCH or head not in TOOL_NODES:
        return ("retrieve", "군 내부 문서를 검색하는 중...")
    if head in TERMINAL_ONLY_TOOLS:
        # 단독 전용 도구(SMALLTALK)는 곧바로 답변 생성이다
        return ("generate", "답변을 생성하는 중...")
    return ("tool", TOOL_STATUS_MESSAGES.get(head, DEFAULT_TOOL_STATUS_MESSAGE))


def _status_after_node(node_name: str, merged_state: dict) -> tuple[str, str] | None:
    """그래프 노드 완료 시 다음 단계 안내 (stage, message). 안내가 없는 노드는 None.

    프론트가 "검색하는 중..." 같은 진행 상태를 표시할 수 있게 하는
    status 이벤트의 재료다. 완료된 노드를 보고 "이제 시작되는 단계"를 알린다.
    """
    if node_name == "route":
        return _next_stage_status(merged_state)
    if node_name in TOOL_NODES and node_name not in TERMINAL_ONLY_TOOLS:
        # 계획 실행 중인 도구 완료 → 남은 큐 기준으로 다음 단계 안내
        return _next_stage_status(merged_state)
    if node_name == "finalize":
        # 복합 계획의 후처리 도구(HWP_EXPORT 등)가 남아 있으면 실행 안내
        pending = merged_state.get("pending_intents") or []
        head = pending[0] if pending else None
        if head in TOOL_NODES:
            return ("tool", TOOL_STATUS_MESSAGES.get(head, DEFAULT_TOOL_STATUS_MESSAGE))
        return None
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
            # LLM 추측이 아니라 실제 적용된 검색 범위를 기록한다
            domain=requested_domain or "ALL",
            sources=[s["name"] for s in sources],
            grounded=grounded,
        )
        async for frame in stream_answer(
            result.get("final_answer") or "", sources, result.get("generated_files") or []
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


@app.get(
    "/health",
    tags=["운영"],
    summary="헬스체크",
    description=(
        "기본은 서버 생존 확인만 한다 (빠름, 모델 상태 미검사).\n\n"
        "`?deep=true`면 의존 서비스 4종을 함께 검사해 집계한다 "
        "(각 검사 timeout 5초 — 장애 시에도 20초 안에 응답):\n\n"
        "- `llm`: OpenAI 호환 `GET {AX_BASE_URL}/models` (vLLM/llama.cpp 공통)\n"
        "- `embedding` / `reranker`: 각 서버의 `GET /health`\n"
        "- `milvus`: 컬렉션 존재 조회\n\n"
        '응답: `{"status": "ok"|"degraded", "services": {이름: {"ok": bool, "detail": str}}}`. '
        "하나라도 실패면 degraded — 미들웨어/프론트의 서버 상태 표시용이며, "
        "degraded여도 /query 요청 자체는 거부되지 않는다."
    ),
)
def health(
    deep: Annotated[
        bool, Query(description="true면 LLM·임베딩·리랭커·Milvus 상태를 함께 검사")
    ] = False,
) -> dict:
    """서버 생존 확인. deep=true면 로컬 의존 서비스 상태를 집계한다."""
    if not deep:
        return {"status": "ok"}
    return check_dependencies()


@app.get("/capabilities", tags=["질의응답"], summary="사용 가능한 domain·tool 목록")
def capabilities() -> dict:
    """POST /query의 domain·tool 필드에 넣을 수 있는 값 목록 (프론트 UI 데이터 소스).

    - `domains`: 검색 범위 한정 값. 검색이 일어날 때만 적용된다
      ("전체 검색"은 domain을 비우거나 "ALL")
    - `tools`: 처리 경로. `forcible=true`인 항목만 tool 필드로 강제 지정 가능.
      tool을 비우면 라우터가 자동 분류한다

    조합 규칙: domain은 "검색하게 될 경우의 범위", tool은 "검색 여부 자체".
    도메인 전용 모드(예: 훈령에서만)는 `tool=DOC_SEARCH` + `domain=DIRECTIVE` 조합.
    """
    return {
        "domains": [{"code": code, "label": DOMAIN_LABELS.get(code, code)} for code in DOMAINS],
        "tools": [
            {
                "code": DOC_SEARCH,
                "description": TOOL_DESCRIPTIONS[DOC_SEARCH],
                "forcible": True,
            },
            *[
                {
                    "code": name,
                    "description": TOOL_DESCRIPTIONS.get(name, ""),
                    "forcible": name in FORCIBLE_TOOLS,
                }
                for name in TOOL_NODES
            ],
        ],
    }


class DocumentItem(BaseModel):
    """적재 문서 1건의 요약 정보."""

    name: str = Field(description="문서 파일명 (source_doc)", examples=["휴가규정.pdf"])
    type: str = Field(description="파일 형식 (확장자 대문자)", examples=["PDF"])
    domain: str = Field(
        description=(
            "적재 시 지정된 도메인 (HR | TECH | FINANCE_LEGAL | GENERAL | MANUAL | DIRECTIVE)"
        ),
        examples=["HR"],
    )
    visibility: str = Field(
        description='공개 범위. "ALL"=전사, "DEPT_ONLY"=소유 부서만 검색 가능',
        examples=["ALL"],
    )
    owning_department: str = Field(
        description=(
            "문서 소유 부서 코드 (적재 시 --department 값). "
            "visibility=DEPT_ONLY일 때 질의자의 user_department와 일치해야 검색된다. "
            "ALL 문서에서는 소유자 기록용 메타데이터"
        ),
        examples=["HR_TEAM"],
    )
    applied_at: datetime = Field(
        description="적재(갱신) 시각 — 청크 중 최신 created_at",
        examples=["2026-07-05T19:09:47"],
    )


class DocumentListOutput(BaseModel):
    """GET /documents 응답 (무한 스크롤 페이지네이션)."""

    documents: list[DocumentItem]
    total: int = Field(description="필터 적용 후 전체 문서 수")
    offset: int
    limit: int
    has_more: bool = Field(description="true면 offset += limit으로 다음 페이지 요청")


@app.get(
    "/documents",
    tags=["문서 관리"],
    response_model=DocumentListOutput,
    summary="적재 문서 목록 (무한 스크롤)",
    description=(
        "벡터스토어에 인덱싱된 문서 목록을 반환한다 (관리·운영용).\n\n"
        "**무한 스크롤**: 첫 요청은 `offset=0`으로 시작하고, 응답의 `has_more`가 "
        "`true`이면 `offset += limit`으로 다음 페이지를 요청한다. 정렬은 문서명 "
        "오름차순으로 고정되어 페이지가 안정적이다.\n\n"
        "**도메인 필터**: `?domain=HR`처럼 지정하면 해당 도메인 문서만 반환한다 "
        "(HR | TECH | FINANCE_LEGAL | GENERAL | MANUAL | DIRECTIVE, 대소문자 무관).\n\n"
        "⚠ DEPT_ONLY 문서의 존재(문서명)도 노출되므로 일반 사용자 화면에 "
        "그대로 내보내지 말 것."
    ),
)
def list_documents(
    offset: Annotated[int, Query(ge=0, description="건너뛸 문서 수")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="한 번에 반환할 문서 수")] = 20,
    domain: Annotated[str | None, Query(description="도메인 필터 (예: HR)")] = None,
) -> DocumentListOutput:
    all_documents = vectorstore.list_documents()
    if domain:
        wanted = domain.strip().upper()
        all_documents = [d for d in all_documents if d["domain"] == wanted]

    total = len(all_documents)
    page = all_documents[offset : offset + limit]
    logger.info(
        "문서 목록 조회: domain=%s, offset=%d, limit=%d → %d/%d건",
        domain or "(전체)",
        offset,
        limit,
        len(page),
        total,
    )
    return DocumentListOutput(
        documents=[
            DocumentItem(
                name=doc["source_doc"],
                type=(
                    doc["source_doc"].rsplit(".", 1)[-1].upper()
                    if "." in doc["source_doc"]
                    else "UNKNOWN"
                ),
                domain=doc["domain"],
                visibility=doc["visibility"],
                owning_department=doc["owning_department"],
                applied_at=datetime.fromtimestamp(doc["applied_at"]),
            )
            for doc in page
        ],
        total=total,
        offset=offset,
        limit=limit,
        has_more=(offset + limit) < total,
    )


# ── 문서 적재/삭제 API (interfaces.md §5) ──────────────────────────────
# 적재는 임베딩 때문에 오래 걸리므로(CPU에서 문서당 수 분) 202 + 백그라운드
# 작업으로 처리하고, 상태는 GET /documents/jobs/{job_id}로 조회한다.

# 업로드 본문 크기 상한 (원문 텍스트 기준 넉넉하게)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Milvus source_doc 필드 길이 제한 (interfaces.md §2: VARCHAR(512))
_SOURCE_DOC_MAX_CHARS = 512

# 백그라운드 적재 작업 레지스트리 (인메모리 — 단일 워커 전제)
_ingest_jobs = IngestJobRegistry()


class IngestJobStatus(BaseModel):
    """적재 작업 1건의 상태 (POST /documents 접수 응답 = 조회 응답)."""

    job_id: str = Field(description="작업 ID (상태 조회 키)")
    status: str = Field(
        description="queued(대기) | running(적재 중) | done(완료) | error(실패)",
        examples=["queued"],
    )
    source_doc: str = Field(description="문서 파일명", examples=["휴가규정.pdf"])
    domain: str
    owning_department: str
    visibility: str
    submitted_at: str | None = Field(description="접수 시각 (ISO)")
    started_at: str | None = Field(description="적재 시작 시각. queued면 null")
    finished_at: str | None = Field(description="종료 시각. 진행 중이면 null")
    chunks_indexed: int | None = Field(description="done일 때 적재된 자식 청크 수")
    deleted_chunks: int | None = Field(description="갱신 적재로 삭제된 기존 청크 수")
    error: str | None = Field(description="error일 때 실패 사유")


class DocumentDeleteOutput(BaseModel):
    """DELETE /documents/{name} 응답."""

    name: str = Field(description="삭제된 문서 파일명")
    deleted_chunks: int = Field(description="삭제된 자식 청크 수")
    deleted_parents: int = Field(description="삭제된 부모 청크 수")


def validate_upload(name: str, domain: str, visibility: str, department: str) -> tuple:
    """업로드 파라미터 검증·정규화. 문제가 있으면 ValueError(한국어 사유).

    파일명은 경로 성분을 제거해 basename만 취한다 (경로 탈출 방지).
    큰따옴표는 Milvus filter 식을 깨뜨리므로 금지한다.
    반환: (safe_name, domain, visibility, department)
    """
    safe_name = (name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not safe_name:
        raise ValueError("문서 파일명(name)이 비어 있다")
    if '"' in safe_name:
        raise ValueError('파일명에 큰따옴표(")는 쓸 수 없다')
    if len(safe_name) > _SOURCE_DOC_MAX_CHARS:
        raise ValueError(f"파일명이 너무 길다 (최대 {_SOURCE_DOC_MAX_CHARS}자)")
    suffix = "." + safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise ValueError(f"지원하지 않는 형식: {safe_name} (지원: {supported})")

    domain_normalized = (domain or "").strip().upper()
    if domain_normalized not in DOMAINS:
        raise ValueError(f"허용되지 않는 domain: {domain!r} (허용: {', '.join(DOMAINS)})")

    visibility_normalized = (visibility or "").strip().upper() or "ALL"
    if visibility_normalized not in ("ALL", "DEPT_ONLY"):
        raise ValueError(f"허용되지 않는 visibility: {visibility!r} (허용: ALL, DEPT_ONLY)")

    department_normalized = (department or "").strip().upper()
    if visibility_normalized == "DEPT_ONLY" and not department_normalized:
        raise ValueError("visibility=DEPT_ONLY에는 department(소유 부서)가 필요하다")
    return safe_name, domain_normalized, visibility_normalized, department_normalized


def _run_ingest_job(job_id: str, path: Path, domain: str, department: str, visibility: str) -> None:
    """백그라운드 적재 실행 (BackgroundTasks가 스레드풀에서 호출).

    적재/삭제는 ingest 모듈 잠금으로 직렬화되므로, 동시에 여러 건이
    접수돼도 한 번에 하나씩 순서대로 처리된다.
    """
    _ingest_jobs.mark_running(job_id)
    try:
        result = ingest.ingest_file(
            path, domain=domain, owning_department=department, visibility=visibility
        )
        _ingest_jobs.mark_done(
            job_id,
            chunks_indexed=result["chunks_indexed"],
            deleted_chunks=result["deleted_children"],
        )
        logger.info(
            "적재 작업 완료: job=%s, 문서=%s → 자식 %d청크",
            job_id,
            path.name,
            result["chunks_indexed"],
        )
    except Exception as exc:
        logger.exception("적재 작업 실패: job=%s, 문서=%s", job_id, path.name)
        _ingest_jobs.mark_error(job_id, f"{exc.__class__.__name__}: {exc}")


@app.post(
    "/documents",
    tags=["문서 관리"],
    status_code=202,
    response_model=IngestJobStatus,
    summary="문서 적재/갱신 (백그라운드)",
    description=(
        "문서 파일을 받아 인덱싱(청킹→임베딩→Milvus 적재→BM25 재빌드)한다.\n\n"
        "**본문은 파일 바이트 그대로** 보낸다 (`Content-Type: application/octet-stream`, "
        "multipart 아님). 파일명·메타데이터는 쿼리 파라미터로 전달한다.\n\n"
        "- 지원 형식: `.md`(섹션 인식) | `.txt` | `.pdf` — 텍스트 인코딩은 "
        "UTF-8·UTF-8 BOM·CP949 자동 인식, 스캔본(이미지) PDF는 실패한다 "
        "(OCR 미지원). 최대 50MB\n"
        "- **같은 `name`이 이미 적재돼 있으면 갱신**: 기존 청크를 지우고 재적재한다\n"
        "- 적재는 임베딩 때문에 오래 걸려(CPU에서 문서당 수 분) **202 + 작업(job)으로 "
        "접수**하고 백그라운드에서 실행한다. 진행 상태는 응답의 `job_id`로 "
        "`GET /documents/jobs/{job_id}` 폴링\n"
        "- 작업은 한 번에 하나만 실행된다 (BM25 전체 재빌드 직렬화). 나머지는 순서 대기\n"
        "- `visibility=DEPT_ONLY`면 `department` 필수\n\n"
        "400: 파라미터 오류(형식·도메인·빈 본문), 413: 50MB 초과"
    ),
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload_document(
    http_request: Request,
    background_tasks: BackgroundTasks,
    name: Annotated[
        str,
        Query(description="문서 파일명 (source_doc). 확장자로 형식을 판별한다"),
    ],
    domain: Annotated[
        str,
        Query(description="문서 도메인: " + " | ".join(DOMAINS)),
    ],
    department: Annotated[
        str,
        Query(description="소유 부서 코드 (ACL). visibility=DEPT_ONLY면 필수"),
    ] = "",
    visibility: Annotated[
        str,
        Query(description='공개 범위: "ALL"(전사, 기본) | "DEPT_ONLY"(소유 부서만)'),
    ] = "ALL",
) -> IngestJobStatus:
    try:
        safe_name, domain_normalized, visibility_normalized, department_normalized = (
            validate_upload(name, domain, visibility, department)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content = await http_request.body()
    if not content:
        raise HTTPException(
            status_code=400, detail="요청 본문이 비어 있다 (파일 바이트를 raw body로 보낼 것)"
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        max_mb = _MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"파일이 너무 크다 (최대 {max_mb}MB)")

    upload_dir = Path(get_config().UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / safe_name
    saved_path.write_bytes(content)

    job = _ingest_jobs.create(
        source_doc=safe_name,
        domain=domain_normalized,
        owning_department=department_normalized,
        visibility=visibility_normalized,
    )
    background_tasks.add_task(
        _run_ingest_job,
        job.job_id,
        saved_path,
        domain_normalized,
        department_normalized,
        visibility_normalized,
    )
    logger.info(
        "적재 작업 접수: job=%s, 문서=%s (%d바이트, domain=%s, visibility=%s)",
        job.job_id,
        safe_name,
        len(content),
        domain_normalized,
        visibility_normalized,
    )
    return IngestJobStatus(**job.to_dict())


@app.get(
    "/documents/jobs",
    tags=["문서 관리"],
    response_model=list[IngestJobStatus],
    summary="최근 적재 작업 목록",
    description=(
        "최근 제출 순(최신 먼저) 적재 작업 목록. "
        "**인메모리 이력**이라 서버 재시작 시 사라진다 (적재된 청크는 유지)."
    ),
)
def list_ingest_jobs(
    limit: Annotated[int, Query(ge=1, le=100, description="반환할 최대 작업 수")] = 20,
) -> list[IngestJobStatus]:
    return [IngestJobStatus(**job.to_dict()) for job in _ingest_jobs.recent(limit)]


@app.get(
    "/documents/jobs/{job_id}",
    tags=["문서 관리"],
    response_model=IngestJobStatus,
    summary="적재 작업 상태 조회",
    description=(
        "POST /documents가 반환한 job_id로 진행 상태를 조회한다 "
        "(queued → running → done | error). 404면 존재하지 않거나 "
        "서버 재시작으로 이력이 사라진 작업이다."
    ),
)
def get_ingest_job(job_id: str) -> IngestJobStatus:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="해당 작업을 찾을 수 없다 (서버 재시작으로 이력이 사라졌을 수 있음)",
        )
    return IngestJobStatus(**job.to_dict())


@app.delete(
    "/documents/{name}",
    tags=["문서 관리"],
    response_model=DocumentDeleteOutput,
    summary="문서 삭제",
    description=(
        "문서의 자식·부모 청크를 전부 삭제하고 BM25 인덱스를 재빌드한다.\n\n"
        "- `name`은 GET /documents가 반환한 문서 파일명 (한글·공백은 URL 인코딩)\n"
        "- **동기 처리**: BM25 전체 재빌드 때문에 수 초~수십 초 걸릴 수 있다\n"
        "- 404: 적재되지 않은 문서, 409: 다른 적재/삭제 작업이 진행 중 (10초 대기 후)\n\n"
        "⚠ 되돌릴 수 없다. 업로드 원본이 UPLOAD_DIR에 남아 있으면 재적재는 가능"
    ),
)
def delete_document(
    name: Annotated[str, PathParam(description="문서 파일명 (예: 휴가규정.pdf)")],
) -> DocumentDeleteOutput:
    decoded = name.strip()
    if not decoded or '"' in decoded:
        raise HTTPException(status_code=400, detail="유효하지 않은 문서 파일명이다")
    try:
        result = ingest.delete_document(decoded)
    except ingest.IngestBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="다른 적재/삭제 작업이 진행 중이다. 잠시 후 다시 시도할 것",
        ) from exc
    if result["deleted_children"] == 0 and result["deleted_parents"] == 0:
        raise HTTPException(status_code=404, detail=f"적재된 문서가 아니다: {decoded}")
    return DocumentDeleteOutput(
        name=decoded,
        deleted_chunks=result["deleted_children"],
        deleted_parents=result["deleted_parents"],
    )


@app.get(
    "/files/{name}",
    tags=["생성 파일"],
    summary="생성 문서 다운로드",
    description=(
        "도구가 생성한 문서 파일(HWPX 등)을 내려받는다.\n\n"
        "- `name`은 SSE `file` 이벤트의 `name` 값 (한글은 URL 인코딩)\n"
        "- 파일은 EXPORT_DIR(기본 ./data/exports)에서만 서빙된다 (경로 탈출 차단)\n"
        "- **임시 보관소**: EXPORT_TTL_HOURS(기본 24시간) 지난 파일은 새 파일 "
        "생성 시점에 자동 정리된다\n"
        "- 404: 존재하지 않는 파일 (TTL 정리로 삭제됐을 수 있음)\n\n"
        "미들웨어는 file 이벤트 수신 즉시 파일을 가져가 자체 저장한다 "
        "(fetch-and-store, interfaces.md §5)."
    ),
)
def download_file(
    name: Annotated[str, PathParam(description="파일명 (예: MARS_답변_20260716_103000.hwpx)")],
) -> FileResponse:
    safe_name = (name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    # 경로 성분이 섞였거나 숨김 파일 형태면 거부한다 (EXPORT_DIR 밖 접근 차단)
    if not safe_name or safe_name != name.strip() or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="유효하지 않은 파일명이다")
    path = Path(get_config().EXPORT_DIR) / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"파일이 없다: {safe_name}")
    return FileResponse(path, filename=safe_name, media_type="application/octet-stream")


_QUERY_RESPONSES = {
    200: {
        "description": (
            "SSE 스트림 (`data: {JSON}\\n\\n` 프레임). 이벤트 순서:\n\n"
            '1. `{"type": "status", "stage": str, "message": str}` — 진행 상태 안내, '
            '0회 이상 (예: "군 내부 문서를 검색하는 중..."). stage 값: '
            "route(질문 분석·계획 수립) | tool(도구 실행, 도구별 문구) | "
            "retrieve | rerank | generate | verify\n"
            '2. `{"type": "text", "content": str}` — 답변 조각, N회 '
            "(문장 단위, STREAM_TEXT_INTERVAL_MS 간격)\n"
            '3. `{"type": "file", "name": str, "url": str, "tool": str}` — '
            "도구가 생성한 파일(HWPX 등), 0회 이상. **미들웨어는 이 이벤트를 "
            "신호로 파일을 가져가 저장**한다 (답변 텍스트 파싱 불필요)\n"
            '4. `{"type": "sources", "items": [{"name": str, "page": null}]}` — '
            "정확히 1회. 근거 검증(verify)을 통과한 답변에만 문서가 담기며, "
            "fallback·잡담 응답은 빈 배열\n"
            '5. `{"type": "done"}` — 종료 신호, 항상 마지막\n\n'
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


@app.post(
    "/query", tags=["질의응답"], summary="질의응답 (SSE 스트리밍)", responses=_QUERY_RESPONSES
)
async def query(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """질의응답 파이프라인을 완주한 뒤 확정 답변을 SSE로 분할 전송한다.

    처리 흐름: 라우팅(잡담 감지·쿼리 재작성) → 하이브리드 검색(벡터+BM25,
    부서 ACL 강제) → 리랭크 → 답변 생성 → 근거 검증(fail-closed) → 스트리밍.

    - 토큰 실시간 스트리밍이 아니라 **검증 통과 후 분할 전송**이다
      (검증 실패 시 재생성되므로, 이미 보낸 텍스트를 취소할 수 없는 SSE 계약과의 정합)
    - `domain`을 지정하면 해당 도메인 문서만 검색한다 (교범=MANUAL, 훈령=DIRECTIVE 등).
      빈 값/"ALL"이면 전 도메인
    - `tool`을 지정하면 라우터 분류 없이 해당 경로로 강제 직행한다 (잡담 예외 없음)
    - `user_department` 누락 시 visibility=ALL 문서만 검색된다
    - **생성 중지**: 별도 API 없음. 클라이언트가 SSE 연결을 중단(abort)하면
      서버가 노드 경계에서 감지해 이후 단계를 실행하지 않는다.
      미들웨어는 프론트 연결 중단 시 본 서버로의 요청도 함께 중단해야 한다
    """
    logger.info(
        "질의 수신: dept=%s, domain=%s, tool=%s, 이력 %d턴, 질문=%s",
        request.user_department or "(없음)",
        request.domain or "(전체)",
        request.tool or "(자동)",
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

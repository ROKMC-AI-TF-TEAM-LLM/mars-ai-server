"""질의응답 엔드포인트: SSE 스트리밍 질의와 요청 가능 값 조회."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ax_rag.api.pipeline import run_pipeline
from ax_rag.api.schemas import QueryRequest
from ax_rag.query_graph.tools import (
    DOC_SEARCH,
    FORCIBLE_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_NODES,
)
from ax_rag.shared.config import DOMAIN_LABELS, DOMAINS
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/capabilities", tags=["질의응답"], summary="사용 가능한 domain·tool 목록")
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


@router.post(
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
        run_pipeline(request, http_request),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # 리버스 프록시 버퍼링 방지 (필수)
            "Cache-Control": "no-cache",
        },
    )

"""운영 엔드포인트: 서버·의존 서비스 상태 점검."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ax_rag.shared.health import check_dependencies

router = APIRouter()


@router.get(
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

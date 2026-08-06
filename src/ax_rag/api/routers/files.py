"""생성 파일 다운로드 엔드포인트 (도구가 만든 HWPX 등)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from ax_rag.shared.config import get_config

router = APIRouter()


@router.get(
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

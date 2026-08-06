"""문서 관리 엔드포인트: 적재·갱신·목록·삭제와 적재 작업 상태 (interfaces.md §5).

적재는 임베딩 때문에 오래 걸리므로(CPU에서 문서당 수 분) 202 + 백그라운드
작업으로 처리하고, 상태는 GET /documents/jobs/{job_id}로 조회한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi import Path as PathParam

from ax_rag.api.normalize import validate_upload
from ax_rag.api.schemas import (
    DocumentDeleteOutput,
    DocumentItem,
    DocumentListOutput,
    IngestJobStatus,
)
from ax_rag.indexer_graph import ingest
from ax_rag.shared import vectorstore
from ax_rag.shared.config import DOMAINS, get_config
from ax_rag.shared.ingest_jobs import IngestJobRegistry
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 업로드 본문 크기 상한 (원문 텍스트 기준 넉넉하게)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# 백그라운드 적재 작업 레지스트리 (인메모리 — 단일 워커 전제)
_ingest_jobs = IngestJobRegistry()


@router.get(
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


@router.post(
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


@router.get(
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


@router.get(
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


@router.delete(
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

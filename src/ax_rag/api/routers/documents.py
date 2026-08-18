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

from ax_rag.api.normalize import validate_project_id_strict, validate_upload
from ax_rag.api.schemas import (
    DocumentDeleteOutput,
    DocumentItem,
    DocumentListOutput,
    IngestJobStatus,
    ProjectDeleteOutput,
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
    project_id: Annotated[
        str | None,
        Query(
            description=(
                '프로젝트 필터. 값을 주면 그 프로젝트 문서만, "" 를 주면 전사 공용 문서만. '
                "생략하면 전부 (관리용이므로 프로젝트 문서도 함께 나온다)"
            )
        ),
    ] = None,
) -> DocumentListOutput:
    all_documents = vectorstore.list_documents()
    if domain:
        wanted = domain.strip().upper()
        all_documents = [d for d in all_documents if d["domain"] == wanted]
    if project_id is not None:
        scope = project_id.strip()
        all_documents = [d for d in all_documents if (d.get("project_id") or "") == scope]

    total = len(all_documents)
    page = all_documents[offset : offset + limit]
    logger.info(
        "문서 목록 조회: domain=%s, project=%s, offset=%d, limit=%d → %d/%d건",
        domain or "(전체)",
        "(전체)" if project_id is None else (project_id or "전사"),
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
                project_id=doc.get("project_id") or "",
                applied_at=datetime.fromtimestamp(doc["applied_at"]),
            )
            for doc in page
        ],
        total=total,
        offset=offset,
        limit=limit,
        has_more=(offset + limit) < total,
    )


def _run_ingest_job(
    job_id: str,
    path: Path,
    domain: str,
    department: str,
    visibility: str,
    project_id: str = "",
) -> None:
    """백그라운드 적재 실행 (BackgroundTasks가 스레드풀에서 호출).

    적재/삭제는 ingest 모듈 잠금으로 직렬화되므로, 동시에 여러 건이
    접수돼도 한 번에 하나씩 순서대로 처리된다.
    """
    _ingest_jobs.mark_running(job_id)
    try:
        result = ingest.ingest_file(
            path,
            domain=domain,
            owning_department=department,
            visibility=visibility,
            project_id=project_id,
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
    project_id: Annotated[
        str,
        Query(
            description=(
                "프로젝트 전용 적재 (선택). 지정하면 그 프로젝트 채팅에서만 검색된다. "
                "빈 값이면 전사 공용. 영숫자·밑줄·하이픈만 허용(최대 64자)하며 "
                "형식 위반은 400 — 조용히 전사 공용이 되면 안 되기 때문이다"
            )
        ),
    ] = "",
) -> IngestJobStatus:
    try:
        (
            safe_name,
            domain_normalized,
            visibility_normalized,
            department_normalized,
            project_normalized,
        ) = validate_upload(name, domain, visibility, department, project_id)
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
        project_normalized,
    )
    logger.info(
        "적재 작업 접수: job=%s, 문서=%s (%d바이트, domain=%s, visibility=%s, project=%s)",
        job.job_id,
        safe_name,
        len(content),
        domain_normalized,
        visibility_normalized,
        project_normalized or "(전사)",
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
        "문서의 자식·부모 청크를 삭제하고 BM25 인덱스를 재빌드한다.\n\n"
        "- 문서 식별자는 **(project_id, name) 복합키**다. 파일명이 같아도 프로젝트가 "
        "다르면 별개 문서이며 서로 영향을 주지 않는다\n"
        "- `project_id`를 생략하면 **전사 공용 문서만** 삭제한다 (프로젝트 문서는 무사)\n"
        "- `name`은 GET /documents가 반환한 문서 파일명 (한글·공백은 URL 인코딩)\n"
        "- **동기 처리**: BM25 전체 재빌드 때문에 수 초~수십 초 걸릴 수 있다\n"
        "- 404: 적재되지 않은 문서, 409: 다른 적재/삭제 작업이 진행 중 (10초 대기 후)\n\n"
        "⚠ 되돌릴 수 없다. 업로드 원본이 UPLOAD_DIR에 남아 있으면 재적재는 가능"
    ),
)
def delete_document(
    name: Annotated[str, PathParam(description="문서 파일명 (예: 휴가규정.pdf)")],
    project_id: Annotated[
        str,
        Query(description='소속 프로젝트. 생략하면 전사 공용 문서("")만 삭제한다'),
    ] = "",
) -> DocumentDeleteOutput:
    decoded = name.strip()
    if not decoded or '"' in decoded:
        raise HTTPException(status_code=400, detail="유효하지 않은 문서 파일명이다")
    scope = ""
    if project_id.strip():
        try:
            scope = validate_project_id_strict(project_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = ingest.delete_document(decoded, project_id=scope)
    except ingest.IngestBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="다른 적재/삭제 작업이 진행 중이다. 잠시 후 다시 시도할 것",
        ) from exc
    if result["deleted_children"] == 0 and result["deleted_parents"] == 0:
        where = f"프로젝트 {scope}" if scope else "전사 공용"
        raise HTTPException(status_code=404, detail=f"{where}에 적재된 문서가 아니다: {decoded}")
    return DocumentDeleteOutput(
        name=decoded,
        project_id=scope,
        deleted_chunks=result["deleted_children"],
        deleted_parents=result["deleted_parents"],
    )


@router.delete(
    "/projects/{project_id}",
    tags=["문서 관리"],
    response_model=ProjectDeleteOutput,
    summary="프로젝트 문서 일괄 삭제",
    description=(
        "프로젝트에 속한 문서를 전부 삭제하고 BM25 인덱스를 한 번 재빌드한다.\n\n"
        "- **전사 공용 문서는 지우지 않는다** (project_id가 빈 문서는 대상이 아니다)\n"
        "- **동기 처리**: BM25 전체 재빌드 때문에 수 초~수십 초 걸릴 수 있다\n"
        "- 404: 그 프로젝트에 적재된 문서가 없음, 409: 다른 적재/삭제 진행 중\n\n"
        "⚠ 되돌릴 수 없다."
    ),
)
def delete_project(
    project_id: Annotated[str, PathParam(description="프로젝트 ID (예: proj-7f3a)")],
) -> ProjectDeleteOutput:
    # 빈 값이면 전사 문서 전체가 대상이 되므로 형식 검증을 엄격히 한다
    try:
        scope = validate_project_id_strict(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = ingest.delete_project(scope)
    except ingest.IngestBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="다른 적재/삭제 작업이 진행 중이다. 잠시 후 다시 시도할 것",
        ) from exc
    if not result["documents"]:
        raise HTTPException(status_code=404, detail=f"적재된 문서가 없는 프로젝트다: {scope}")
    return ProjectDeleteOutput(
        project_id=scope,
        documents=result["documents"],
        deleted_chunks=result["deleted_children"],
        deleted_parents=result["deleted_parents"],
    )

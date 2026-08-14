"""API 요청·응답 모델 (미들웨어 계약, interfaces.md §5).

Field description은 Swagger UI에 그대로 노출되는 연동 문서다 — 미들웨어
개발자가 읽는 1차 자료이므로 코드 변경 시 함께 갱신할 것.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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
    project_id: str = Field(
        default="",
        description=(
            "프로젝트 채팅 범위 (선택). 지정하면 **전사 문서 + 그 프로젝트 문서**를 "
            "검색한다. 빈 값이면 전사 문서만 검색하며 프로젝트 문서는 노출되지 않는다. "
            "영숫자·밑줄·하이픈만 허용(최대 64자)하고, 형식 위반은 전사 검색으로 처리한다. "
            "⚠️ **이 값의 멤버십 검증은 미들웨어 책임이다** — MARS는 받은 값을 신뢰한다 "
            "(user_department와 동일 신뢰 모델)"
        ),
        examples=["proj-7f3a"],
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

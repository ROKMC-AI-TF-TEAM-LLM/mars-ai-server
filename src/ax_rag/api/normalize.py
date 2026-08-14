"""요청 파라미터 정규화·검증 (신뢰 경계).

미들웨어가 보낸 값을 그래프·적재 파이프라인이 쓸 내부 표현으로 바꾼다.
질의 파라미터(domain/tool)는 관대하게(미지의 값은 무시하고 기본 경로로),
적재 파라미터(validate_upload)는 엄격하게(오류면 400) 처리한다 —
질의는 실패해도 검색 범위가 넓어질 뿐이지만, 적재는 잘못된 메타데이터가
영구히 남기 때문이다.
"""

from __future__ import annotations

import re

from ax_rag.indexer_graph.loaders import SUPPORTED_SUFFIXES
from ax_rag.query_graph.tools import DOC_SEARCH, FORCIBLE_TOOLS, TOOL_NODES
from ax_rag.shared.config import DOMAINS
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# Milvus source_doc 필드 길이 제한 (interfaces.md §2: VARCHAR(512))
_SOURCE_DOC_MAX_CHARS = 512

# Milvus project_id 필드 길이 제한 (interfaces.md §2: VARCHAR(64))
_PROJECT_ID_MAX_CHARS = 64

# project_id 허용 문자 — acl._sanitize와 같은 집합.
# 여기서 미리 거르면 "정제되어 조용히 다른 프로젝트가 되는" 상황을 없앤다
_PROJECT_ID_ALLOWED = re.compile(r"^[0-9A-Za-z_\-]+$")


def normalize_project_id(value: str) -> str:
    """요청 project_id 정규화. 빈 값 또는 형식 위반이면 ""(전사 검색).

    ⚠️ 이 값의 **멤버십 검증은 미들웨어 책임**이다 (user_department와 동일
    신뢰 모델). 여기서는 형식만 본다 — 형식이 어긋나면 프로젝트 문서를
    보여주지 않는 쪽(전사만)으로 떨어뜨린다 (fail-closed).
    """
    normalized = (value or "").strip()
    if not normalized:
        return ""
    if len(normalized) > _PROJECT_ID_MAX_CHARS:
        logger.warning("project_id가 너무 길다 (%d자) → 전사 검색", len(normalized))
        return ""
    if not _PROJECT_ID_ALLOWED.match(normalized):
        logger.warning("project_id 형식 위반: %r → 전사 검색", value)
        return ""
    return normalized


def validate_project_id_strict(value: str) -> str:
    """project_id 엄격 검증. 비었거나 형식 위반이면 ValueError(한국어 사유).

    적재·삭제처럼 **틀리면 되돌릴 수 없는** 경로에서 쓴다. 질의(normalize_project_id)와
    달리 조용히 ""로 떨어뜨리지 않는다 — 적재에서는 프로젝트 문서가 전사 공용이 되고,
    삭제에서는 전사 문서 전체가 대상이 된다.
    """
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("project_id가 비어 있다")
    if len(normalized) > _PROJECT_ID_MAX_CHARS:
        raise ValueError(f"project_id가 너무 길다 (최대 {_PROJECT_ID_MAX_CHARS}자)")
    if not _PROJECT_ID_ALLOWED.match(normalized):
        raise ValueError(f"허용되지 않는 project_id: {value!r} (영숫자·밑줄·하이픈만)")
    return normalized


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


def validate_upload(
    name: str, domain: str, visibility: str, department: str, project_id: str = ""
) -> tuple[str, str, str, str, str]:
    """업로드 파라미터 검증·정규화. 문제가 있으면 ValueError(한국어 사유).

    파일명은 경로 성분을 제거해 basename만 취한다 (경로 탈출 방지).
    큰따옴표는 Milvus filter 식을 깨뜨리므로 금지한다.

    ⚠️ project_id는 질의와 달리 **엄격하게** 본다. 질의에서 형식 위반은 전사
    검색으로 떨어져 범위가 좁아지지만, 적재에서 ""로 떨어지면 프로젝트 전용
    문서가 **전사 공용으로 영구히 남는다** — 조용히 넘기면 안 된다.

    반환: (safe_name, domain, visibility, department, project_id)
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

    # 빈 값(전사 공용)은 허용하고, 값이 있으면 엄격 검증
    project_normalized = (
        validate_project_id_strict(project_id) if (project_id or "").strip() else ""
    )
    return (
        safe_name,
        domain_normalized,
        visibility_normalized,
        department_normalized,
        project_normalized,
    )

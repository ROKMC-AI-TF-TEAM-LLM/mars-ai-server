"""ACL·프로젝트 필터 (CLAUDE.md 보안 규칙).

- dense 검색: build_acl_filter_expr()로 Milvus 스칼라 필터 표현식을 만들어 적용
- BM25 검색: 별도 인덱스라 Milvus 필터가 못 미치므로 filter_by_acl() 후처리 필수.
  이 필터를 우회하는 검색 경로를 만들지 않는다

★ 두 함수는 **같은 판정을 해야 한다.** 한쪽만 고치면 그 검색 축으로만 문서가
새어 나가고, dense 결과만 보면 정상이라 눈에 띄지 않는다.

정책:
- visibility "ALL"은 전 부서 열람 가능, "DEPT_ONLY"는 소유 부서만
- user_department 누락 시 가장 제한적으로: visibility ALL만 (interfaces.md §5)
- domain이 GENERAL이면 도메인 제한 없이 전 도메인 검색
- project_id "" 문서는 전사 공용(부서 ACL 적용), 값이 있으면 그 프로젝트 전용.
  프로젝트 전용 문서의 접근 통제는 project_id가 담당한다 (visibility는 ALL 전제)
- 알 수 없는 visibility 값은 배제한다 (fail-closed)
"""

from __future__ import annotations

import re

# Milvus 표현식 인젝션 방지: 값에 허용하는 문자 (한글/영숫자/밑줄/하이픈)
_UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z_\-가-힣]")


def _sanitize(value: str) -> str:
    """표현식에 삽입되는 값에서 허용 문자 외를 제거한다 (인젝션 방지)."""
    return _UNSAFE_CHARS.sub("", value or "")


def build_acl_filter_expr(domain: str, user_department: str, project_id: str = "") -> str:
    """dense 검색용 Milvus 스칼라 필터 표현식을 만든다.

    project_id가 비면 전사 문서만, 값이 있으면 전사 + 그 프로젝트를 검색한다.
    프로젝트는 범위를 **자기 문서에 한해서만** 넓힌다 — 전사 문서의 부서 ACL은
    프로젝트 안에서도 그대로 걸린다.

    project_id가 정제 후 비게 되면(허용 문자 없음) 전사 문서만 남는다 —
    범위가 좁아지는 방향이라 fail-closed다.
    """
    department = _sanitize(user_department)
    safe_domain = _sanitize(domain)
    safe_project = _sanitize(project_id)

    if department:
        acl = (
            f'(visibility == "ALL" or '
            f'(visibility == "DEPT_ONLY" and owning_department == "{department}"))'
        )
    else:
        # user_department 누락: DEPT_ONLY 전부 배제 (가장 제한적 폴백)
        acl = 'visibility == "ALL"'

    scope = f'(project_id == "" and {acl})'
    if safe_project:
        scope = f'({scope} or project_id == "{safe_project}")'

    if safe_domain and safe_domain != "GENERAL":
        return f'domain == "{safe_domain}" and {scope}'
    return scope


def filter_by_acl(
    candidates: list[dict], domain: str, user_department: str, project_id: str = ""
) -> list[dict]:
    """BM25 결과에 ACL·프로젝트 후처리 필터 적용. 우회 금지.

    build_acl_filter_expr와 **같은 판정**이어야 한다. 후보에 project_id
    메타데이터가 없으면 전사 공용("")으로 취급되므로, BM25 corpus가 그 필드를
    반드시 실어야 한다 (indexer_graph.graph._BM25_META_FIELDS).
    """
    department = user_department or ""
    scope_project = project_id or ""
    allowed: list[dict] = []
    for candidate in candidates:
        if domain and domain != "GENERAL" and candidate.get("domain") != domain:
            continue

        candidate_project = candidate.get("project_id") or ""
        if candidate_project:
            # 프로젝트 전용 문서: 같은 프로젝트일 때만 (접근 통제를 project_id가 담당)
            if candidate_project == scope_project:
                allowed.append(candidate)
            continue

        # 전사 공용 문서: 기존 부서 ACL을 그대로 적용
        visibility = candidate.get("visibility")
        if visibility == "ALL":
            allowed.append(candidate)
        elif (
            visibility == "DEPT_ONLY"
            and department
            and candidate.get("owning_department") == department
        ):
            allowed.append(candidate)
        # 그 외(값 누락/미지의 값)는 fail-closed로 배제
    return allowed

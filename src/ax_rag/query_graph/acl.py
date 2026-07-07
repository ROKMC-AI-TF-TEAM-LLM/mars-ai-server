"""ACL 필터 (CLAUDE.md 보안 규칙).

- dense 검색: build_acl_filter_expr()로 Milvus 스칼라 필터 표현식을 만들어 적용
- BM25 검색: 별도 인덱스라 Milvus 필터가 못 미치므로 filter_by_acl() 후처리 필수.
  이 필터를 우회하는 검색 경로를 만들지 않는다

정책:
- visibility "ALL"은 전 부서 열람 가능, "DEPT_ONLY"는 소유 부서만
- user_department 누락 시 가장 제한적으로: visibility ALL만 (interfaces.md §5)
- domain이 GENERAL이면 도메인 제한 없이 전 도메인 검색
- 알 수 없는 visibility 값은 배제한다 (fail-closed)
"""

from __future__ import annotations

import re

# Milvus 표현식 인젝션 방지: 값에 허용하는 문자 (한글/영숫자/밑줄/하이픈)
_UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z_\-가-힣]")


def _sanitize(value: str) -> str:
    """표현식에 삽입되는 값에서 허용 문자 외를 제거한다 (인젝션 방지)."""
    return _UNSAFE_CHARS.sub("", value or "")


def build_acl_filter_expr(domain: str, user_department: str) -> str:
    """dense 검색용 Milvus 스칼라 필터 표현식을 만든다."""
    department = _sanitize(user_department)
    safe_domain = _sanitize(domain)

    if department:
        acl = (
            f'(visibility == "ALL" or '
            f'(visibility == "DEPT_ONLY" and owning_department == "{department}"))'
        )
    else:
        # user_department 누락: DEPT_ONLY 전부 배제 (가장 제한적 폴백)
        acl = 'visibility == "ALL"'

    if safe_domain and safe_domain != "GENERAL":
        return f'domain == "{safe_domain}" and {acl}'
    return acl


def filter_by_acl(candidates: list[dict], domain: str, user_department: str) -> list[dict]:
    """BM25 결과에 ACL 후처리 필터 적용. dense는 Milvus 필터로
    처리되지만 bm25는 별도 인덱스라 코드에서 걸러야 함. 우회 금지."""
    department = user_department or ""
    allowed: list[dict] = []
    for candidate in candidates:
        if domain and domain != "GENERAL" and candidate.get("domain") != domain:
            continue
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

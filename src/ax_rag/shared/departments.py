"""부서 코드 공통 검증. 잘못된 식별자를 다른 부서로 변환하지 않는다."""

import re

_DEPARTMENT = re.compile(r"[0-9A-Za-z_가-힣-]+")


def validate_department(value: str) -> str:
    """빈 부서는 허용하고, 내부 공백·특수문자·VARCHAR 바이트 초과는 거부한다."""
    normalized = value.strip().upper()
    if normalized and (
        not _DEPARTMENT.fullmatch(normalized) or len(normalized.encode("utf-8")) > 32
    ):
        raise ValueError("부서 코드는 한글·영숫자·밑줄·하이픈만 허용하며 최대 32바이트다")
    return normalized

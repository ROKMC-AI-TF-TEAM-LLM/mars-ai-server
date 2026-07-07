"""main.py 문서 업로드 파라미터 검증 유닛 테스트 (validate_upload)."""

from __future__ import annotations

import main
import pytest


def test_정상_업로드_파라미터_정규화() -> None:
    assert main.validate_upload("휴가규정.md", "hr", "all", "hr_team") == (
        "휴가규정.md",
        "HR",
        "ALL",
        "HR_TEAM",
    )


def test_경로_성분은_제거하고_basename만_쓴다() -> None:
    """경로 탈출(디렉터리 트래버설) 방지 — 업로드 디렉터리 밖에 쓰지 않는다."""
    name, _, _, _ = main.validate_upload("..\\..\\비밀문서.md", "HR", "ALL", "")
    assert name == "비밀문서.md"
    name, _, _, _ = main.validate_upload("../etc/훈령.pdf", "DIRECTIVE", "ALL", "")
    assert name == "훈령.pdf"


def test_visibility_기본값은_ALL() -> None:
    _, _, visibility, _ = main.validate_upload("문서.txt", "GENERAL", "", "")
    assert visibility == "ALL"


def test_지원하지_않는_확장자는_거부() -> None:
    with pytest.raises(ValueError, match="지원하지 않는 형식"):
        main.validate_upload("한글문서.hwp", "HR", "ALL", "")
    with pytest.raises(ValueError, match="지원하지 않는 형식"):
        main.validate_upload("확장자없음", "HR", "ALL", "")


def test_미지의_domain은_거부() -> None:
    """검색 필터(normalize_requested_domain)와 달리 적재는 엄격하다."""
    with pytest.raises(ValueError, match="허용되지 않는 domain"):
        main.validate_upload("문서.md", "MARKETING", "ALL", "")
    with pytest.raises(ValueError, match="허용되지 않는 domain"):
        main.validate_upload("문서.md", "", "ALL", "")


def test_DEPT_ONLY는_department가_필수() -> None:
    with pytest.raises(ValueError, match="department"):
        main.validate_upload("인사기록.md", "HR", "DEPT_ONLY", "")
    # department가 있으면 통과
    _, _, visibility, department = main.validate_upload("인사기록.md", "HR", "DEPT_ONLY", "HR_TEAM")
    assert (visibility, department) == ("DEPT_ONLY", "HR_TEAM")


def test_큰따옴표가_든_파일명은_거부() -> None:
    """Milvus filter 식(source_doc == \"...\")을 깨뜨리는 문자."""
    with pytest.raises(ValueError, match="큰따옴표"):
        main.validate_upload('규정".md', "HR", "ALL", "")


def test_빈_파일명은_거부() -> None:
    with pytest.raises(ValueError, match="비어 있다"):
        main.validate_upload("   ", "HR", "ALL", "")


def test_미지의_visibility는_거부() -> None:
    with pytest.raises(ValueError, match="visibility"):
        main.validate_upload("문서.md", "HR", "PUBLIC", "")

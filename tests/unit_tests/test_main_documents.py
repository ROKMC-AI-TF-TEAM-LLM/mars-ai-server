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
        "",  # project_id 미지정 = 전사 공용
    )


def test_경로_성분은_제거하고_basename만_쓴다() -> None:
    """경로 탈출(디렉터리 트래버설) 방지 — 업로드 디렉터리 밖에 쓰지 않는다."""
    name = main.validate_upload("..\\..\\비밀문서.md", "HR", "ALL", "")[0]
    assert name == "비밀문서.md"
    name = main.validate_upload("../etc/훈령.pdf", "DIRECTIVE", "ALL", "")[0]
    assert name == "훈령.pdf"


def test_visibility_기본값은_ALL() -> None:
    assert main.validate_upload("문서.txt", "GENERAL", "", "")[2] == "ALL"


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
    _, _, visibility, department, _ = main.validate_upload(
        "인사기록.md", "HR", "DEPT_ONLY", "HR_TEAM"
    )
    assert (visibility, department) == ("DEPT_ONLY", "HR_TEAM")


# ---------- project_id (적재는 엄격, 질의는 관대) ----------


def test_적재_project_id는_그대로_보존된다() -> None:
    assert main.validate_upload("부대지침.md", "HR", "ALL", "", "proj-7f3a")[4] == "proj-7f3a"


def test_적재_project_id_형식_위반은_거부한다() -> None:
    """★ 질의와 달리 조용히 ""로 떨어뜨리면 안 된다 —
    프로젝트 전용 문서가 전사 공용으로 영구히 남는다."""
    for bad in ['proj" or 1==1', "proj a", "프로젝트", "p" * 65]:
        with pytest.raises(ValueError, match="project_id"):
            main.validate_upload("문서.md", "HR", "ALL", "", bad)


def test_질의_project_id는_형식_위반이면_전사로_떨어진다() -> None:
    """검색은 범위가 좁아지는 방향이라 관대해도 안전하다 (fail-closed)."""
    from ax_rag.api.normalize import normalize_project_id

    assert normalize_project_id("proj-7f3a") == "proj-7f3a"
    assert normalize_project_id("") == ""
    assert normalize_project_id('" or project_id != "') == ""
    assert normalize_project_id("프로젝트") == ""
    assert normalize_project_id("p" * 65) == ""


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

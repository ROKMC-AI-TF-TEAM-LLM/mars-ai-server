"""업로드 스테이징 경로 — 같은 파일명을 다른 프로젝트가 덮어쓰지 않는가.

적재는 백그라운드 작업이라 파일 저장 시점과 실제 적재 시점이 떨어져 있다.
경로가 파일명뿐이면 그 사이에 다른 프로젝트 업로드가 파일을 갈아치워,
나중에 실행되는 작업이 **남의 내용을 자기 프로젝트로 적재**한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ax_rag.api import create_app
from ax_rag.shared.config import get_config


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_config.cache_clear()
    # 실제 적재(임베딩·Milvus)는 돌리지 않는다 — 스테이징 경로만 본다
    from ax_rag.api.routers import documents as documents_module

    monkeypatch.setattr(documents_module, "_run_ingest_job", lambda *a, **kw: None)
    yield TestClient(create_app())
    get_config.cache_clear()


def _upload(client: TestClient, body: bytes, project_id: str = "") -> None:
    params = {"name": "휴가규정.md", "domain": "HR", "department": "HR_TEAM"}
    if project_id:
        params["project_id"] = project_id
    response = client.post("/documents", params=params, content=body)
    assert response.status_code == 202, response.text


def test_같은_이름을_다른_프로젝트가_덮어쓰지_않는다(client: TestClient, tmp_path: Path) -> None:
    """★ 이 격리가 깨지면 적재 작업이 남의 문서 내용을 자기 프로젝트로 넣는다."""
    _upload(client, "전사 내용".encode(), project_id="")
    _upload(client, "A 부대 내용".encode(), project_id="proj-a")
    _upload(client, "B 부대 내용".encode(), project_id="proj-b")

    uploads = tmp_path / "uploads"
    assert (uploads / "휴가규정.md").read_text(encoding="utf-8") == "전사 내용"
    assert (uploads / "proj-a" / "휴가규정.md").read_text(encoding="utf-8") == "A 부대 내용"
    assert (uploads / "proj-b" / "휴가규정.md").read_text(encoding="utf-8") == "B 부대 내용"


def test_같은_프로젝트_재업로드는_덮어쓴다(client: TestClient, tmp_path: Path) -> None:
    """같은 (project_id, name)은 동일 문서이므로 갱신이 맞다."""
    _upload(client, "구버전".encode(), project_id="proj-a")
    _upload(client, "신버전".encode(), project_id="proj-a")
    staged = tmp_path / "uploads" / "proj-a" / "휴가규정.md"
    assert staged.read_text(encoding="utf-8") == "신버전"


def test_전사_업로드는_기존_경로_그대로다(client: TestClient, tmp_path: Path) -> None:
    """하위 호환 — 프로젝트를 안 쓰던 기존 문서는 UPLOAD_DIR 바로 아래에 남는다."""
    _upload(client, "전사 내용".encode())
    assert (tmp_path / "uploads" / "휴가규정.md").is_file()

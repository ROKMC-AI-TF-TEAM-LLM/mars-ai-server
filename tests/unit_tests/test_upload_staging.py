"""동명 업로드의 작업별 본문·ACL 격리와 종료 후 스테이징 보존 검증."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ax_rag.api import create_app
from ax_rag.api.routers import documents
from ax_rag.shared.config import get_config


@pytest.fixture()
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_config.cache_clear()
    queued: list[tuple] = []
    # 실제 작업은 대기시킨다. 두 번째 접수 뒤 첫 번째 작업의 경로를 읽는다.
    monkeypatch.setattr(documents, "_run_ingest_job", lambda *args: queued.append(args))
    yield TestClient(create_app()), queued
    get_config.cache_clear()


@pytest.mark.parametrize("projects", [("", ""), ("proj-a", "proj-a"), ("proj-a", "proj-b")])
def test_대기중인_작업의_본문과_ACL은_다음_업로드로_바뀌지_않는다(staging, projects) -> None:
    client, queued = staging
    for text, visibility, project in zip(
        ["공개 본문", "비공개 본문"], ["ALL", "DEPT_ONLY"], projects, strict=True
    ):
        response = client.post(
            "/documents",
            params={
                "name": "휴가규정.md",
                "domain": "HR",
                "department": "HR_TEAM",
                "visibility": visibility,
                "project_id": project,
            },
            content=text.encode("utf-8"),
        )
        assert response.status_code == 202
    first, second = queued
    assert first[1] != second[1]
    assert first[1].name == second[1].name == "휴가규정.md"
    assert (first[1].read_text(encoding="utf-8"), first[4]) == ("공개 본문", "ALL")
    assert (second[1].read_text(encoding="utf-8"), second[4]) == ("비공개 본문", "DEPT_ONLY")
    assert first[5] == projects[0] and second[5] == projects[1]


@pytest.mark.parametrize("fails", [False, True])
def test_종료한_작업의_원본을_자동_삭제하지_않는다(tmp_path, monkeypatch, fails) -> None:
    task_dir = tmp_path / "ingest-job"
    task_dir.mkdir()
    path = task_dir / "휴가규정.md"
    path.write_text("본문", encoding="utf-8")
    other = tmp_path / "다른작업.md"
    other.write_text("보존", encoding="utf-8")

    def ingest(*args, **kwargs):
        if fails:
            raise TimeoutError("임베딩 실패")
        return {"chunks_indexed": 1, "deleted_children": 0}

    monkeypatch.setattr(documents.ingest, "ingest_file", ingest)
    job = documents._ingest_jobs.create(path.name, "HR", "HR_TEAM", "ALL")
    documents._run_ingest_job(job.job_id, path, "HR", "HR_TEAM", "ALL")
    assert job.status == ("error" if fails else "done")
    assert path.read_text(encoding="utf-8") == "본문"
    assert other.read_text(encoding="utf-8") == "보존"

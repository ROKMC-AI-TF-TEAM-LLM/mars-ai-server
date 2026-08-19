"""프로젝트 일괄 삭제 유닛 테스트 — 전사 문서를 지우지 않는 것이 핵심이다."""

from __future__ import annotations

import pytest

from ax_rag.api.normalize import validate_project_id_strict
from ax_rag.indexer_graph import ingest as ingest_module
from ax_rag.indexer_graph.ingest import delete_project
from ax_rag.shared import vectorstore

_DOCUMENTS = [
    {"source_doc": "휴가규정.md", "project_id": ""},  # 전사 공용
    {"source_doc": "부대지침.md", "project_id": "proj-a"},
    {"source_doc": "야간비행.md", "project_id": "proj-a"},
    {"source_doc": "타부대.md", "project_id": "proj-b"},
]


class _Recorder:
    def __init__(self) -> None:
        self.child_filter: str | None = None
        self.parent_scopes: list[tuple[str, str]] = []
        self.deleted_parent_ids: list[str] = []
        self.rebuilt = 0

    def delete_by_project(self, project_id: str) -> int:
        self.child_filter = project_id
        return 7

    def fetch_parent_ids(self, source_doc: str, project_id: str = "") -> list[str]:
        self.parent_scopes.append((source_doc, project_id))
        return [f"p-{source_doc}"]

    def delete_parents(self, parent_ids: list[str]) -> int:
        self.deleted_parent_ids = list(parent_ids)
        return len(parent_ids)

    def rebuild(self) -> None:
        self.rebuilt += 1


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(ingest_module.vectorstore, "list_documents", lambda: list(_DOCUMENTS))
    monkeypatch.setattr(ingest_module.vectorstore, "delete_by_project", rec.delete_by_project)
    monkeypatch.setattr(ingest_module.vectorstore, "fetch_parent_ids", rec.fetch_parent_ids)
    monkeypatch.setattr(ingest_module.parent_store, "delete_by_parent_ids", rec.delete_parents)
    monkeypatch.setattr(ingest_module, "rebuild_bm25", rec.rebuild)
    return rec


def test_해당_프로젝트_문서만_지운다(recorder: _Recorder) -> None:
    result = delete_project("proj-a")
    assert result["documents"] == ["부대지침.md", "야간비행.md"]
    assert recorder.child_filter == "proj-a"
    # 부모 참조 수집도 프로젝트로 스코프된다 — 전사·타 프로젝트 부모는 건드리지 않는다
    assert recorder.parent_scopes == [("부대지침.md", "proj-a"), ("야간비행.md", "proj-a")]
    assert recorder.deleted_parent_ids == ["p-부대지침.md", "p-야간비행.md"]


def test_BM25는_문서마다가_아니라_한_번만_재빌드한다(recorder: _Recorder) -> None:
    delete_project("proj-a")
    assert recorder.rebuilt == 1


def test_문서가_없는_프로젝트는_아무것도_지우지_않는다(recorder: _Recorder) -> None:
    result = delete_project("proj-none")
    assert result["documents"] == []
    assert result["deleted_children"] == 0
    assert recorder.child_filter is None  # 삭제 호출 자체가 없다
    assert recorder.rebuilt == 0


def test_빈_project_id는_거부한다(recorder: _Recorder) -> None:
    """★ 빈 값을 허용하면 전사 문서 전체가 삭제 대상이 된다."""
    with pytest.raises(ValueError, match="project_id"):
        delete_project("")
    assert recorder.child_filter is None


def test_저장소_계층도_빈_project_id를_거부한다() -> None:
    """호출부를 우회해 직접 불러도 막힌다 (이중 방어)."""
    with pytest.raises(ValueError, match="project_id"):
        vectorstore.delete_by_project("")


def test_엄격_검증은_형식_위반을_거부한다() -> None:
    assert validate_project_id_strict("proj-7f3a") == "proj-7f3a"
    for bad in ["", "  ", 'proj" or 1==1', "proj a", "프로젝트", "p" * 65]:
        with pytest.raises(ValueError, match="project_id"):
            validate_project_id_strict(bad)

"""indexer_graph/ingest.py 유닛 테스트 — 저장소·그래프를 가짜로 대체해 순서/잠금 계약 검증."""

from __future__ import annotations

from pathlib import Path

import pytest

from ax_rag.indexer_graph import ingest as ingest_module
from ax_rag.indexer_graph.ingest import IngestBusyError, delete_document, ingest_file


class _CallRecorder:
    """호출 순서와 인자를 기록하는 가짜 저장소/그래프."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.invoked_state: dict | None = None
        self.children_deleted = 3
        self.parents_deleted = 1

    def fake_load(self, path: Path) -> tuple[str, list[dict] | None]:
        self.calls.append("load")
        return "제1조 휴가는 연 21일로 한다.", None

    def fake_delete_children(self, source_doc: str) -> int:
        self.calls.append("delete_children")
        return self.children_deleted

    def fake_delete_parents(self, source_doc: str) -> int:
        self.calls.append("delete_parents")
        return self.parents_deleted

    def fake_invoke(self, state: dict) -> dict:
        self.calls.append("invoke")
        self.invoked_state = state
        return {"chunks_indexed": 7}

    def fake_rebuild(self) -> None:
        self.calls.append("rebuild_bm25")


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _CallRecorder:
    rec = _CallRecorder()
    monkeypatch.setattr(ingest_module, "load_document", rec.fake_load)
    monkeypatch.setattr(ingest_module.vectorstore, "delete_by_source_doc", rec.fake_delete_children)
    monkeypatch.setattr(ingest_module.parent_store, "delete_by_source_doc", rec.fake_delete_parents)
    monkeypatch.setattr(ingest_module.graph, "invoke", rec.fake_invoke)
    monkeypatch.setattr(ingest_module, "rebuild_bm25", rec.fake_rebuild)
    return rec


# ---------- ingest_file ----------


def test_적재는_추출_검증_후_삭제_재적재_순서다(recorder: _CallRecorder) -> None:
    result = ingest_file(Path("휴가규정.md"), "HR", "HR_TEAM", "ALL")
    # 추출 실패 시 기존 데이터를 지우지 않기 위해 load가 반드시 삭제보다 먼저다
    assert recorder.calls == ["load", "delete_children", "delete_parents", "invoke"]
    assert result == {
        "source_doc": "휴가규정.md",
        "deleted_children": 3,
        "deleted_parents": 1,
        "chunks_indexed": 7,
    }


def test_적재_상태에_메타데이터가_그대로_전달된다(recorder: _CallRecorder) -> None:
    ingest_file(Path("국방훈령.pdf"), "DIRECTIVE", "HQ", "DEPT_ONLY")
    state = recorder.invoked_state or {}
    assert state["source_doc"] == "국방훈령.pdf"
    assert state["domain"] == "DIRECTIVE"
    assert state["owning_department"] == "HQ"
    assert state["visibility"] == "DEPT_ONLY"
    assert "휴가는 연 21일" in state["text"]


def test_텍스트_추출_실패면_삭제_없이_ValueError(
    recorder: _CallRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def empty_load(path: Path) -> tuple[str, None]:
        recorder.calls.append("load")
        return "   ", None  # 스캔본 PDF처럼 공백만 추출된 경우

    monkeypatch.setattr(ingest_module, "load_document", empty_load)
    with pytest.raises(ValueError, match="추출하지 못했다"):
        ingest_file(Path("스캔본.pdf"), "HR", "HR_TEAM", "ALL")
    assert recorder.calls == ["load"]  # 기존 데이터 삭제가 일어나지 않았다


# ---------- delete_document ----------


def test_삭제_후_BM25를_재빌드한다(recorder: _CallRecorder) -> None:
    result = delete_document("휴가규정.md")
    assert recorder.calls == ["delete_children", "delete_parents", "rebuild_bm25"]
    assert result == {
        "source_doc": "휴가규정.md",
        "deleted_children": 3,
        "deleted_parents": 1,
    }


def test_미적재_문서는_BM25_재빌드를_건너뛴다(recorder: _CallRecorder) -> None:
    recorder.children_deleted = 0
    recorder.parents_deleted = 0
    result = delete_document("없는문서.md")
    assert "rebuild_bm25" not in recorder.calls
    assert result["deleted_children"] == 0


def test_잠금_경합이면_IngestBusyError(recorder: _CallRecorder) -> None:
    """다른 적재가 진행 중(잠금 보유)이면 기다렸다가 예외를 낸다."""
    assert ingest_module._LOCK.acquire(timeout=1)  # 진행 중인 적재를 흉내
    try:
        with pytest.raises(IngestBusyError):
            delete_document("휴가규정.md", wait_seconds=0.05)
    finally:
        ingest_module._LOCK.release()
    assert recorder.calls == []  # 잠금을 못 얻으면 아무것도 삭제하지 않는다

"""실패 주입으로 문서 갱신 전후 데이터와 복구 저널 계약을 검증한다."""

from copy import deepcopy
from pathlib import Path

import pytest

from ax_rag.indexer_graph import graph as graph_module
from ax_rag.indexer_graph import ingest
from ax_rag.shared.config import get_config
from ax_rag.shared.store_guard import ensure_store_ready, recovery_path


class Store:
    """부분 쓰기 실패도 재현하는 인메모리 저장소."""

    def __init__(self) -> None:
        self.children = [
            {
                "chunk_id": "old",
                "parent_id": "old-parent",
                "text": "기존 규정",
                "source_doc": "휴가규정.md",
                "project_id": "proj-a",
                "embedding": [0.0] * 1024,
                "domain": "HR",
                "owning_department": "HR_TEAM",
                "visibility": "ALL",
                "chunk_index": 0,
                "created_at": 1,
                "doc_classification": "NORMAL",
            },
            {
                "chunk_id": "other",
                "parent_id": "other-parent",
                "text": "타 부대 규정",
                "source_doc": "휴가규정.md",
                "project_id": "proj-b",
                "embedding": [0.0] * 1024,
                "domain": "HR",
                "owning_department": "HR_TEAM",
                "visibility": "ALL",
                "chunk_index": 0,
                "created_at": 1,
                "doc_classification": "NORMAL",
            },
        ]
        self.parents = {"old-parent": "기존 규정", "other-parent": "타 부대 규정"}
        self.bm25 = deepcopy(self.children)
        self.failure = ""
        self.broken = False
        self.mutations = 0

    def fail(self, stage: str) -> None:
        if self.broken or stage == self.failure:
            self.failure = ""
            raise RuntimeError(f"{stage} 저장 실패")

    def fetch(self, name: str, project: str) -> list[dict]:
        return deepcopy(
            [c for c in self.children if (c["source_doc"], c["project_id"]) == (name, project)]
        )

    def delete(self, name: str, project: str = "") -> int:
        self.mutations += 1
        self.fail("delete")
        old = self.fetch(name, project)
        self.children = [c for c in self.children if c not in old]
        return len(old)

    def delete_parents(self, ids: list[str]) -> int:
        count = sum(pid in self.parents for pid in ids)
        for pid in ids:
            self.parents.pop(pid, None)
        return count

    def insert_parents(self, rows: list[dict]) -> int:
        self.parents.update({p["parent_id"]: p["parent_text"] for p in rows})
        self.fail("parents")
        return len(rows)

    def insert(self, rows: list[dict]) -> int:
        self.children.extend(deepcopy(rows))
        self.fail("children")
        return len(rows)

    def rebuild(self) -> None:
        self.bm25 = deepcopy(self.children)
        self.fail("bm25")


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25"))
    get_config.cache_clear()
    db = Store()
    monkeypatch.setattr(ingest, "load_document", lambda path: ("새 규정은 휴가가 20일이다.", None))
    monkeypatch.setattr(graph_module, "_embed_texts", lambda texts: [[1.0] * 1024 for _ in texts])
    monkeypatch.setattr(ingest.vectorstore, "fetch_document_children", db.fetch)
    monkeypatch.setattr(
        ingest.vectorstore,
        "fetch_parent_ids",
        lambda name, project: [r["parent_id"] for r in db.fetch(name, project)],
    )
    monkeypatch.setattr(ingest.vectorstore, "delete_by_source_doc", db.delete)
    monkeypatch.setattr(ingest.vectorstore, "insert_children", db.insert)
    monkeypatch.setattr(ingest.parent_store, "get_parent", lambda pid: db.parents.get(pid, ""))
    monkeypatch.setattr(ingest.parent_store, "delete_by_parent_ids", db.delete_parents)
    monkeypatch.setattr(ingest.parent_store, "insert_parents", db.insert_parents)
    monkeypatch.setattr(ingest, "rebuild_bm25", db.rebuild)
    yield db
    get_config.cache_clear()


def update() -> dict:
    return ingest.ingest_file(Path("휴가규정.md"), "HR", "HR_TEAM", "DEPT_ONLY", "proj-a")


def test_갱신은_같은_프로젝트만_교체한다(store: Store) -> None:
    other = deepcopy(store.fetch("휴가규정.md", "proj-b"))
    result = update()
    assert result["deleted_children"] == result["deleted_parents"] == 1
    mine = store.fetch("휴가규정.md", "proj-a")
    assert len(mine) == result["chunks_indexed"] > 0
    assert all("20일" in row["text"] and row["visibility"] == "DEPT_ONLY" for row in mine)
    assert store.fetch("휴가규정.md", "proj-b") == other
    assert store.parents["other-parent"] == "타 부대 규정"
    assert "old-parent" not in store.parents
    assert store.bm25 == store.children
    assert not recovery_path().exists()


def test_임베딩_실패는_삭제나_저널을_만들지_않는다(store, monkeypatch) -> None:
    before = deepcopy(store.__dict__)

    def fail(texts):
        raise TimeoutError("임베딩 타임아웃")

    monkeypatch.setattr(graph_module, "_embed_texts", fail)
    with pytest.raises(TimeoutError):
        update()
    assert store.__dict__ == before
    assert not recovery_path().exists()


def test_추출_실패도_기존_데이터를_보존한다(store, monkeypatch) -> None:
    monkeypatch.setattr(ingest, "load_document", lambda path: ("   ", None))
    with pytest.raises(ValueError, match="추출"):
        update()
    assert store.mutations == 0


@pytest.mark.parametrize("stage", ["delete", "parents", "children", "bm25"])
def test_부분_저장_실패도_본문_ACL_임베딩_부모_BM25를_복구한다(store, stage) -> None:
    before_children, before_parents = deepcopy((store.children, store.parents))
    store.failure = stage
    with pytest.raises(RuntimeError, match="저장 실패"):
        update()
    assert sorted(store.children, key=lambda r: r["chunk_id"]) == sorted(
        before_children, key=lambda r: r["chunk_id"]
    )
    assert store.parents == before_parents
    assert store.bm25 == store.children
    assert not recovery_path().exists()
    ensure_store_ready()


def test_복구까지_실패하면_저널로_차단하고_이후_재실행한다(store, monkeypatch) -> None:
    before = deepcopy(store.fetch("휴가규정.md", "proj-a"))

    def fail_after_delete(rows):
        store.broken = True
        raise RuntimeError("DB 연결 중단")

    monkeypatch.setattr(ingest.vectorstore, "insert_children", fail_after_delete)
    with pytest.raises(RuntimeError, match="DB 연결"):
        update()
    assert recovery_path().exists()
    with pytest.raises(RuntimeError, match="미완료"):
        ensure_store_ready()
    with pytest.raises(RuntimeError, match="미완료"):
        update()
    # 프로세스 메모리의 실패 플래그 없이도 디스크 저널만으로 복구한다.
    store.broken = False
    monkeypatch.setattr(ingest.vectorstore, "insert_children", store.insert)
    assert ingest.recover_pending_ingest()
    assert store.fetch("휴가규정.md", "proj-a") == before
    assert store.bm25 == store.children
    assert not ingest.recover_pending_ingest()


def test_신규_문서_실패도_부분_부모를_남기지_않는다(store) -> None:
    before = deepcopy((store.children, store.parents))
    store.failure = "parents"
    with pytest.raises(RuntimeError):
        ingest.ingest_file(Path("새문서.md"), "HR", "HR_TEAM", "ALL")
    assert (store.children, store.parents) == before
    assert not recovery_path().exists()


def test_삭제는_프로젝트를_한정하고_BM25를_갱신한다(store) -> None:
    result = ingest.delete_document("휴가규정.md", "proj-a")
    assert result["deleted_children"] == result["deleted_parents"] == 1
    assert store.fetch("휴가규정.md", "proj-b")
    assert store.bm25 == store.children


def test_잠금_경합이면_데이터_변경없이_거부한다(store) -> None:
    assert ingest._LOCK.acquire(timeout=1)
    try:
        with pytest.raises(ingest.IngestBusyError):
            ingest.delete_document("휴가규정.md", wait_seconds=0.01)
    finally:
        ingest._LOCK.release()
    assert store.mutations == 0

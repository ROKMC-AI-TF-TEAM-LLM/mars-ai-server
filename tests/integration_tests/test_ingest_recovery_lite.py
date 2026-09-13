"""임시 Milvus Lite에서 실제 스냅샷·복구·BM25 정합성 검증 (모델 서버 불필요)."""

import importlib
from pathlib import Path

import pytest

from ax_rag.indexer_graph import ingest
from ax_rag.shared import bm25_store, parent_store, vectorstore
from ax_rag.shared.config import get_config
from ax_rag.shared.store_guard import ensure_store_ready, recovery_path

pytestmark = pytest.mark.integration


@pytest.fixture()
def isolated_lite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MILVUS_LITE_PATH", str(tmp_path / "milvus.db"))
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25"))
    get_config.cache_clear()
    vectorstore.get_client.cache_clear()
    bm25_store._clear_cache()
    graph_module = importlib.import_module("ax_rag.indexer_graph.graph")
    monkeypatch.setattr(graph_module, "_embed_texts", lambda texts: [[1.0] * 1024 for _ in texts])
    path = tmp_path / "휴가규정.md"
    path.write_text("연차 휴가는 매년 15일 부여한다.", encoding="utf-8")
    try:
        ingest.ingest_file(path, "HR", "HR_TEAM", "ALL", "proj-a")
        yield path
    finally:
        vectorstore.get_client().close()
        vectorstore.get_client.cache_clear()
        bm25_store._clear_cache()
        get_config.cache_clear()


def test_실제_Lite_부분삽입_실패를_복구한다(isolated_lite, monkeypatch) -> None:
    path = isolated_lite
    old = vectorstore.fetch_document_children(path.name, "proj-a")
    assert old and len(old[0]["embedding"]) == 1024
    before_bm25 = bm25_store.bm25_search("연차 휴가")
    path.write_text("연차 휴가는 매년 20일 부여한다.", encoding="utf-8")
    insert = vectorstore.insert_children
    failed = False

    def insert_then_fail(rows):
        nonlocal failed
        result = insert(rows)
        if not failed:
            failed = True
            raise RuntimeError("삽입 후 연결 중단")
        return result

    monkeypatch.setattr(vectorstore, "insert_children", insert_then_fail)
    with pytest.raises(RuntimeError, match="삽입 후"):
        ingest.ingest_file(path, "HR", "HR_TEAM", "DEPT_ONLY", "proj-a")
    restored = vectorstore.fetch_document_children(path.name, "proj-a")
    assert restored == old
    assert parent_store.get_parent(old[0]["parent_id"]) == "연차 휴가는 매년 15일 부여한다."
    assert bm25_store.bm25_search("연차 휴가") == before_bm25
    assert not recovery_path().exists()


def test_실제_Lite_재시작_후_저널로_복구한다(isolated_lite, monkeypatch) -> None:
    path = isolated_lite
    before = vectorstore.fetch_document_children(path.name, "proj-a")
    path.write_text("새 규정은 20일이다.", encoding="utf-8")
    insert = vectorstore.insert_children

    def stopped(rows):
        insert(rows)
        # Exception 처리부를 건너뛰는 프로세스 중단을 흉내 낸다.
        raise SystemExit("프로세스 중단")

    with monkeypatch.context() as patcher:
        patcher.setattr(vectorstore, "insert_children", stopped)
        with pytest.raises(SystemExit):
            ingest.ingest_file(path, "HR", "HR_TEAM", "DEPT_ONLY", "proj-a")
    vectorstore.get_client().close()
    vectorstore.get_client.cache_clear()
    bm25_store._clear_cache()
    with pytest.raises(RuntimeError, match="미완료"):
        ensure_store_ready()
    assert ingest.recover_pending_ingest()
    assert vectorstore.fetch_document_children(path.name, "proj-a") == before
    assert all(r["visibility"] == "ALL" for r in bm25_store.bm25_search("연차 휴가"))
    ensure_store_ready()

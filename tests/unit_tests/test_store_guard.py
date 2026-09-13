"""저널이 남은 저장소에서는 검색이 실행되지 않으며, 교체 도중 검색도 대기한다."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ax_rag.query_graph import retrieval
from ax_rag.shared.config import get_config
from ax_rag.shared.store_guard import STORE_LOCK, recovery_path


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25"))
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_미완료_저널이_있으면_검색서비스를_호출하지_않는다(guard, monkeypatch) -> None:
    path = recovery_path()
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")

    def must_not_run(state):
        pytest.fail("복구 전 검색이 실행됨")

    monkeypatch.setattr(retrieval, "dense_retrieve", must_not_run)
    with pytest.raises(RuntimeError, match="미완료"):
        retrieval.run_search({"question": "휴가"}, "휴가")


def test_검색은_교체_잠금이_풀린_뒤에_실행된다(guard, monkeypatch) -> None:
    started, called = threading.Event(), threading.Event()

    def dense(state):
        called.set()
        return {}

    monkeypatch.setattr(retrieval, "dense_retrieve", dense)
    for name in ("bm25_retrieve", "fuse", "rerank"):
        monkeypatch.setattr(retrieval, name, lambda state: {})

    def search():
        started.set()
        return retrieval.run_search({"question": "휴가"}, "휴가")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with STORE_LOCK:
            future = pool.submit(search)
            assert started.wait(2)
            assert not called.wait(0.05)
        assert future.result(timeout=2) == []
    assert called.is_set()


def test_검색끼리는_동시에_실행할_수_있다(guard, monkeypatch) -> None:
    arrived = threading.Barrier(2)

    def dense(state):
        arrived.wait(timeout=2)
        return {}

    monkeypatch.setattr(retrieval, "dense_retrieve", dense)
    for name in ("bm25_retrieve", "fuse", "rerank"):
        monkeypatch.setattr(retrieval, name, lambda state: {})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(retrieval.run_search, {"question": "휴가"}, "휴가") for _ in range(2)
        ]
        assert [f.result(timeout=3) for f in futures] == [[], []]

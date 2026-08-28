"""컬렉션 load 보장 — released 상태면 search/query가 거부된다.

Milvus Lite 2.4.11은 자동 로드해 줘서 그동안 호출이 없어도 문제가 없었다.
그러나 Milvus 서버는 재시작 후 released가 되고, milvus-lite 3.x도 서버와 같은
시맨틱을 따른다. 실측(WSL, 3.2.1)에서 적재 4배치 중 3개가 이걸로 실패했다.
docs/milvus_lite_3x.md 참조.
"""

from __future__ import annotations

from typing import Any

import pytest

from ax_rag.shared import vectorstore


class _FakeClient:
    """load 관련 호출만 기록하는 최소 스텁."""

    def __init__(self, state: str | None = "NotLoad", raise_on_state: bool = False) -> None:
        self._state = state
        self._raise_on_state = raise_on_state
        self.loaded: list[str] = []

    def get_load_state(self, collection_name: str) -> dict[str, Any]:
        if self._raise_on_state:
            raise RuntimeError("이 버전에는 get_load_state가 없다")
        return {"state": self._state}

    def load_collection(self, name: str) -> None:
        self.loaded.append(name)


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch):
    def _apply(client: _FakeClient) -> _FakeClient:
        monkeypatch.setattr(vectorstore, "get_client", lambda: client)
        return client

    return _apply


def test_released면_load를_호출한다(patch_client) -> None:
    """★ 이걸 빠뜨리면 새 프로세스의 첫 조회가 통째로 실패한다."""
    client = patch_client(_FakeClient(state="NotLoad"))
    assert vectorstore.ensure_loaded("company_docs") == "company_docs"
    assert client.loaded == ["company_docs"]


def test_이미_로드됐으면_load를_다시_부르지_않는다(patch_client) -> None:
    """질의마다 불리는 경로라 불필요한 왕복을 만들지 않는다."""
    client = patch_client(_FakeClient(state="Loaded"))
    vectorstore.ensure_loaded("company_docs")
    assert client.loaded == []


def test_상태_조회가_실패하면_load를_시도한다(patch_client) -> None:
    """get_load_state가 없는 구버전에서도 동작해야 한다."""
    client = patch_client(_FakeClient(raise_on_state=True))
    vectorstore.ensure_loaded("company_docs")
    assert client.loaded == ["company_docs"]


def test_load가_실패해도_예외를_전파하지_않는다(patch_client) -> None:
    """load를 지원하지 않는 환경에서도 검색 자체는 시도해 봐야 한다.

    모듈 로거는 propagate=False라 caplog가 보지 못한다 — 직접 핸들러를 단다.
    """
    import logging

    class _NoLoadClient(_FakeClient):
        def load_collection(self, name: str) -> None:
            raise RuntimeError("load 미지원")

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    vectorstore.logger.addHandler(handler)
    try:
        patch_client(_NoLoadClient(state="NotLoad"))
        assert vectorstore.ensure_loaded("company_docs") == "company_docs"
    finally:
        vectorstore.logger.removeHandler(handler)

    assert any("load 실패" in r.getMessage() for r in records)


def test_부모_컬렉션도_로드를_보장한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """부모 치환 조회도 같은 이유로 실패한다 — 실측에서 document_parents가 걸렸다."""
    from ax_rag.shared import parent_store

    client = _FakeClient(state="NotLoad")
    monkeypatch.setattr(parent_store, "get_client", lambda: client)
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    monkeypatch.setattr(client, "has_collection", lambda name: True, raising=False)

    assert parent_store.get_parent_collection() == parent_store.PARENT_COLLECTION
    assert client.loaded == [parent_store.PARENT_COLLECTION]

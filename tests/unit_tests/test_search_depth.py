"""검색 깊이 설정 배선 테스트.

SEARCH_TOP_K(dense/bm25 검색 깊이)와 RERANK_TOP_K(fuse가 리랭커에
넘길 후보 수)가 하드코딩이 아니라 config에서 읽히는지 검증한다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from ax_rag.query_graph.nodes import bm25_retrieve as bm25_module
from ax_rag.query_graph.nodes import dense_retrieve as dense_module
from ax_rag.query_graph.nodes.fuse import fuse
from ax_rag.shared.config import get_config


@pytest.fixture()
def set_env_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """환경변수를 설정하고 config 캐시를 재로드한다. 종료 시 캐시를 비워 누수를 막는다."""

    def _set(**kwargs: object) -> None:
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
        get_config.cache_clear()

    yield _set
    get_config.cache_clear()


def _bm25_result(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"연차휴가 규정 조항 {chunk_id}",
        "source_doc": "휴가규정.pdf",
        "parent_id": f"p_{chunk_id}",
        "domain": "HR",
        "owning_department": "HR_TEAM",
        "visibility": "ALL",
        "bm25_score": 1.0,
    }


def test_bm25는_SEARCH_TOP_K로_오버샘플_검색_후_절단한다(
    monkeypatch: pytest.MonkeyPatch, set_env_config: Callable[..., None]
) -> None:
    set_env_config(SEARCH_TOP_K=4)
    captured: dict[str, int] = {}

    def fake_search(query: str, top_k: int) -> list[dict]:
        captured["top_k"] = top_k
        return [_bm25_result(f"c{i}") for i in range(top_k)]

    monkeypatch.setattr(bm25_module, "bm25_search", fake_search)
    result = bm25_module.bm25_retrieve(
        {"question": "연차휴가", "requested_domain": "", "user_department": "HR_TEAM"}
    )

    assert captured["top_k"] == 4 * 3  # 오버샘플 배수 유지
    assert len(result["bm25_candidates"]) == 4  # ACL 후 SEARCH_TOP_K로 절단


def test_dense는_SEARCH_TOP_K를_limit으로_넘긴다(
    monkeypatch: pytest.MonkeyPatch, set_env_config: Callable[..., None]
) -> None:
    set_env_config(SEARCH_TOP_K=7)
    captured: dict[str, int] = {}

    def fake_search(name: str, data: list, filter: str, limit: int, output_fields: list) -> list:  # noqa: A002
        captured["limit"] = limit
        return [[]]

    fake_client = type("FakeClient", (), {"search": staticmethod(fake_search)})()
    monkeypatch.setattr(dense_module, "_embed_query", lambda q: [0.0] * 1024)
    monkeypatch.setattr(dense_module, "get_client", lambda: fake_client)
    monkeypatch.setattr(dense_module, "get_collection", lambda: "company_docs")

    dense_module.dense_retrieve(
        {"question": "연차휴가", "requested_domain": "", "user_department": "HR_TEAM"}
    )
    assert captured["limit"] == 7


def test_fuse는_RERANK_TOP_K개만_확정한다(set_env_config: Callable[..., None]) -> None:
    set_env_config(RERANK_TOP_K=3)
    dense = [
        {"chunk_id": f"c{i}", "text": f"육아휴직 관련 청크 {i}", "dense_score": 1.0 - i * 0.1}
        for i in range(10)
    ]

    result = fuse({"dense_candidates": dense, "bm25_candidates": []})
    assert len(result["retrieved_candidates"]) == 3

"""BM25 검색 깊이 — 프로젝트 지정 시 후처리 필터가 살아남을 만큼 깊이 뽑는가.

BM25는 Milvus 필터가 못 미쳐 후처리로만 거를 수 있다. 프로젝트 문서가 코퍼스에서
극소수면 얕게 뽑은 상위권에 하나도 없어 BM25 축이 통째로 죽는다.
"""

from __future__ import annotations

import pytest

from ax_rag.query_graph.nodes import bm25_retrieve as module
from ax_rag.query_graph.nodes.bm25_retrieve import _OVERSAMPLE_FACTOR, _search_depth


def test_프로젝트가_없으면_기존_오버샘플_그대로다() -> None:
    """기존 동작 무변경 — 회귀 방지선."""
    assert _search_depth(20, "") == 20 * _OVERSAMPLE_FACTOR


def test_프로젝트_지정_시_코퍼스_전체를_훑는다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "corpus_size", lambda: 1525)
    assert _search_depth(20, "proj-a") == 1525


def test_코퍼스가_작으면_기본_오버샘플보다_얕아지지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """인덱스가 비었거나(0) 작을 때 오히려 얕게 뽑는 역전을 막는다."""
    monkeypatch.setattr(module, "corpus_size", lambda: 0)
    assert _search_depth(20, "proj-a") == 20 * _OVERSAMPLE_FACTOR


def test_상한을_넘지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """코퍼스가 커지면 메모리 폭주를 막는다 (그 지점이 프로젝트별 인덱스 전환 신호)."""
    monkeypatch.setattr(module, "corpus_size", lambda: 10**6)
    assert _search_depth(20, "proj-a") == module._PROJECT_SEARCH_DEPTH_CAP

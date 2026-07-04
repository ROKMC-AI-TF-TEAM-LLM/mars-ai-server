"""retrieval_graph/fusion.py 유닛 테스트 — RRF 순위 결합 정확성 (roadmap 3단계 DoD)."""

from __future__ import annotations

import pytest

from ax_rag.retrieval_graph.fusion import rrf_fuse


def _mk(chunk_id: str, **extra: object) -> dict:
    return {"chunk_id": chunk_id, "text": f"본문 {chunk_id}", **extra}


def test_양쪽_리스트에_있는_청크가_최상위로_온다() -> None:
    dense = [_mk("공통"), _mk("dense1"), _mk("dense2")]
    bm25 = [_mk("bm25a"), _mk("공통")]
    fused = rrf_fuse(dense, bm25, k=60)
    assert fused[0]["chunk_id"] == "공통"


def test_rrf_점수_계산이_정확하다() -> None:
    dense = [_mk("공통")]  # dense 1위
    bm25 = [_mk("단독"), _mk("공통")]  # bm25 1위 단독, 2위 공통
    fused = rrf_fuse(dense, bm25, k=60)
    by_id = {c["chunk_id"]: c["rrf_score"] for c in fused}
    assert by_id["공통"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["단독"] == pytest.approx(1 / 61)


def test_bm25가_비면_dense_단독_순서가_유지된다() -> None:
    """bm25 인덱스 부재 폴백: dense 순위가 그대로 결과가 된다."""
    dense = [_mk("1위"), _mk("2위"), _mk("3위")]
    fused = rrf_fuse(dense, [], k=60)
    assert [c["chunk_id"] for c in fused] == ["1위", "2위", "3위"]


def test_top_n으로_잘린다() -> None:
    dense = [_mk(f"d{i}") for i in range(30)]
    fused = rrf_fuse(dense, [], top_n=20)
    assert len(fused) == 20


def test_양쪽_필드가_병합된다() -> None:
    dense = [_mk("공통", dense_score=0.9, parent_id="p1", source_doc="a.md")]
    bm25 = [_mk("공통", bm25_score=3.2)]
    fused = rrf_fuse(dense, bm25)
    item = fused[0]
    assert item["dense_score"] == 0.9
    assert item["bm25_score"] == 3.2
    assert item["parent_id"] == "p1"
    assert item["source_doc"] == "a.md"


def test_빈_입력은_빈_결과() -> None:
    assert rrf_fuse([], []) == []

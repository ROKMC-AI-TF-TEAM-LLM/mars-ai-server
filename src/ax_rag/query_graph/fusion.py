"""RRF(Reciprocal Rank Fusion) 융합 (architecture.md §4)."""

from __future__ import annotations


def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_n: int = 20,
) -> list[dict]:
    """Reciprocal Rank Fusion. k는 관례값 60에서 시작, 평가로 조정.

    각 결과 리스트에서 순위 r(1부터)에 대해 1/(k+r)를 chunk_id별로 합산해
    내림차순 상위 top_n개를 반환한다. 두 리스트에 모두 등장한 청크는
    필드가 병합되며(dense_score, bm25_score 공존), rrf_score가 추가된다.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}
    for results in (dense_results, bm25_results):
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            merged.setdefault(chunk_id, {}).update(item)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [{**merged[chunk_id], "rrf_score": score} for chunk_id, score in ordered]

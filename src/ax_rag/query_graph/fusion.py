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


def merge_query_results(per_query: list[list[dict]], top_n: int) -> list[dict]:
    """쿼리별 융합 결과를 가중 쿼터로 병합한다 (chunk_id 중복 제거).

    대표 쿼리(첫 항목)가 자리의 **최소 절반**을 갖고, 나머지를 부쿼리들이
    균등 분배한다. 남는 자리는 순위대로 백필한다.

    균등 분배가 아닌 이유: RERANK_TOP_K가 작을 때(개발 환경 10) 3쿼리 균등이면
    대표 쿼리 몫이 10 → 3으로 줄어 **기존에 잘 찾던 문서가 밀려난다**. 다중 쿼리는
    측면을 넓히려는 것이지 대표 쿼리를 희생하려는 것이 아니다.
    """
    if not per_query:
        return []
    if len(per_query) == 1:
        return per_query[0][:top_n]

    merged: list[dict] = []
    seen: set[str] = set()

    def take(items: list[dict], limit: int) -> None:
        """중복을 건너뛰며 limit개까지 담는다."""
        taken = 0
        for item in items:
            if taken >= limit or len(merged) >= top_n:
                return
            chunk_id = item["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(item)
            taken += 1

    primary_quota = max(1, (top_n + 1) // 2)
    sub_count = len(per_query) - 1
    sub_quota = max(1, (top_n - primary_quota) // sub_count)

    take(per_query[0], primary_quota)
    for sub in per_query[1:]:
        take(sub, sub_quota)
    for items in per_query:
        take(items, top_n)
    return merged[:top_n]

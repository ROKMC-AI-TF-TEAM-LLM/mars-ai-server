"""다중 쿼리 검색 테스트 — 쿼리별 검색과 가중 쿼터 병합.

배경: 단일 쿼리는 하나의 의미 이웃만 긁어 다면적 질문에서 근거가 한쪽으로
쏠린다 (실측: "조건·기간·신청 방법"을 물었는데 근거 청크 1건).
쿼리를 나눠 검색하되 **대표 쿼리 몫을 지켜** 기존에 잘 찾던 문서가 밀려나지
않게 한다.

ReAct 전환 후 한 번의 run_search는 쿼리 하나만 쓰고, 여러 측면은 에이전트가
**라운드를 나눠** 검색해 근거를 누적한다. 아래 노드들의 다중 쿼리 지원은
그대로 두었다 — 평가 스크립트가 쓰고, 향후 한 라운드 안에서 다면 검색을
할 때 재사용된다.
"""

from __future__ import annotations

import pytest

from ax_rag.query_graph.fusion import merge_query_results
from ax_rag.query_graph.nodes import bm25_retrieve as bm25_module
from ax_rag.query_graph.nodes import dense_retrieve as dense_module
from ax_rag.query_graph.nodes.fuse import fuse
from ax_rag.shared.config import get_config

# ── 쿼리별 검색 ────────────────────────────────────────────────────────────


class _FakeClient:
    """Milvus search 계약만 흉내 낸다. 호출마다 다른 chunk_id를 돌려준다."""

    def __init__(self) -> None:
        self.call_count = 0

    def search(self, name: str, data: list, filter: str, limit: int, output_fields: list) -> list:  # noqa: A002
        self.call_count += 1
        return [
            [
                {
                    "chunk_id": f"q{self.call_count}_c{i}",
                    "distance": 0.9 - i * 0.1,
                    "entity": {field: "값" for field in output_fields},
                }
                for i in range(2)
            ]
        ]


def test_dense는_쿼리마다_검색하고_query_index를_남긴다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(dense_module, "_embed_queries", lambda qs: [[0.0] * 1024 for _ in qs])
    monkeypatch.setattr(dense_module, "get_client", lambda: fake)
    monkeypatch.setattr(dense_module, "get_collection", lambda: "company_docs")

    result = dense_module.dense_retrieve(
        {
            "question": "휴가",
            "rewritten_query": "휴가규정",
            "search_queries": ["휴가규정", "휴가 신청 절차"],
            "requested_domain": "",
            "user_department": "HR_TEAM",
        }
    )
    assert fake.call_count == 2  # 쿼리마다 Milvus 검색
    indices = {c["query_index"] for c in result["dense_candidates"]}
    assert indices == {0, 1}


def test_임베딩은_쿼리가_여럿이어도_1회_호출한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """임베딩 서버가 texts 배열을 받으므로 다중 쿼리의 지연 비용이 거의 없다."""
    calls: list[list[str]] = []

    def fake_embed(queries: list[str]) -> list[list[float]]:
        calls.append(queries)
        return [[0.0] * 1024 for _ in queries]

    monkeypatch.setattr(dense_module, "_embed_queries", fake_embed)
    monkeypatch.setattr(dense_module, "get_client", lambda: _FakeClient())
    monkeypatch.setattr(dense_module, "get_collection", lambda: "company_docs")

    dense_module.dense_retrieve(
        {
            "question": "휴가",
            "search_queries": ["가", "나", "다"],
            "requested_domain": "",
            "user_department": "HR_TEAM",
        }
    )
    assert len(calls) == 1 and calls[0] == ["가", "나", "다"]


def _bm25_row(chunk_id: str, visibility: str = "ALL", dept: str = "HR_TEAM") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"본문 {chunk_id}",
        "source_doc": "휴가규정.md",
        "parent_id": f"p_{chunk_id}",
        "domain": "HR",
        "owning_department": dept,
        "visibility": visibility,
        "bm25_score": 1.0,
    }


def test_bm25는_쿼리마다_ACL을_적용한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 보안: ACL 후처리를 우회하는 경로를 만들지 않는다 (쿼리별 적용)."""
    monkeypatch.setattr(
        bm25_module,
        "bm25_search",
        lambda q, top_k: [_bm25_row("ok"), _bm25_row("secret", "DEPT_ONLY", "FIN_TEAM")],
    )
    result = bm25_module.bm25_retrieve(
        {
            "question": "휴가",
            "search_queries": ["휴가규정", "휴가 신청 절차"],
            "requested_domain": "",
            "user_department": "HR_TEAM",
        }
    )
    ids = [c["chunk_id"] for c in result["bm25_candidates"]]
    assert "secret" not in ids  # 타 부서 DEPT_ONLY는 쿼리 수와 무관하게 배제
    assert {c["query_index"] for c in result["bm25_candidates"]} == {0, 1}


# ── 융합: 가중 쿼터 병합 ───────────────────────────────────────────────────


def _cand(chunk_id: str, query_index: int = 0) -> dict:
    return {"chunk_id": chunk_id, "text": f"본문 {chunk_id}", "query_index": query_index}


def test_단일_쿼리면_기존_융합_경로를_그대로_탄다() -> None:
    """회귀 방지: 쿼리가 1개일 때 결과가 종전과 같아야 한다."""
    dense = [_cand(f"c{i}") for i in range(5)]
    result = fuse({"dense_candidates": dense, "bm25_candidates": []})
    assert [c["chunk_id"] for c in result["retrieved_candidates"]] == [f"c{i}" for i in range(5)]


def test_대표_쿼리가_자리의_절반_이상을_갖는다() -> None:
    primary = [_cand(f"p{i}") for i in range(10)]
    secondary = [_cand(f"s{i}") for i in range(10)]
    merged = merge_query_results([primary, secondary], top_n=10)
    primary_taken = sum(1 for c in merged if c["chunk_id"].startswith("p"))
    assert primary_taken >= 5  # 대표 쿼리 몫 보장
    assert len(merged) == 10


def test_병합은_중복을_제거한다() -> None:
    shared = _cand("공통")
    merged = merge_query_results([[shared, _cand("a")], [shared, _cand("b")]], top_n=10)
    ids = [c["chunk_id"] for c in merged]
    assert ids.count("공통") == 1
    assert set(ids) == {"공통", "a", "b"}


def test_한쪽이_비면_남은_쪽으로_자리를_채운다() -> None:
    primary = [_cand(f"p{i}") for i in range(8)]
    merged = merge_query_results([primary, []], top_n=6)
    assert len(merged) == 6  # 백필로 자리를 다 채운다


def test_빈_입력은_빈_결과() -> None:
    assert merge_query_results([], top_n=5) == []


def test_fuse가_쿼리별로_융합해_병합한다() -> None:
    """쿼리별 RRF → 가중 쿼터 병합. 전체를 한 번에 RRF하지 않는다."""
    get_config.cache_clear()
    top_k = get_config().RERANK_TOP_K
    dense = [_cand(f"q0_{i}", 0) for i in range(top_k)] + [
        _cand(f"q1_{i}", 1) for i in range(top_k)
    ]
    result = fuse({"dense_candidates": dense, "bm25_candidates": []})
    candidates = result["retrieved_candidates"]
    assert len(candidates) == top_k
    # 두 쿼리 모두 최종 후보에 기여한다 (측면 커버리지)
    assert {c["query_index"] for c in candidates} == {0, 1}

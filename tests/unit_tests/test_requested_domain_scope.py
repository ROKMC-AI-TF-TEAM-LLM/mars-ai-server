"""요청 기반 도메인 한정 검색 테스트.

검색 필터의 도메인은 요청(requested_domain)이 유일한 출처다:
- 미지정("") → 도메인 무관 검색 (단, 보안 필터 visibility/부서는 항상 적용)
- 지정(HR 등) → 해당 도메인만
라우터의 LLM 분류(state["domain"])는 검색 범위에 영향을 주지 않아야 한다.
"""

from __future__ import annotations

import pytest

from ax_rag.query_graph.nodes import bm25_retrieve as bm25_module
from ax_rag.query_graph.nodes import dense_retrieve as dense_module

_BM25_RESULTS = [
    {
        "chunk_id": "hr1",
        "text": "국가를 당사자로 하는 계약에 관한 법률 ...",
        "source_doc": "국가계약법.pdf",
        "parent_id": "p1",
        "domain": "HR",  # 실측 시나리오: 법령 문서가 HR로 적재됨
        "owning_department": "HR_TEAM",
        "visibility": "ALL",
        "bm25_score": 3.2,
    },
    {
        "chunk_id": "fin1",
        "text": "경비 정산 규정 ...",
        "source_doc": "경비규정.pdf",
        "parent_id": "p2",
        "domain": "FINANCE_LEGAL",
        "owning_department": "FIN_TEAM",
        "visibility": "ALL",
        "bm25_score": 2.5,
    },
    {
        "chunk_id": "secret",
        "text": "타 부서 전용 문서",
        "source_doc": "비밀.pdf",
        "parent_id": "p3",
        "domain": "HR",
        "owning_department": "FIN_TEAM",
        "visibility": "DEPT_ONLY",
        "bm25_score": 2.0,
    },
]


def test_bm25_도메인_미지정이면_전_도메인_검색_보안은_유지(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bm25_module, "bm25_search", lambda q, top_k: list(_BM25_RESULTS))
    result = bm25_module.bm25_retrieve(
        {
            "question": "국가 계약",
            "domain": "FINANCE_LEGAL",  # 라우터 분류 — 검색에 영향 없어야 함
            "requested_domain": "",
            "user_department": "HR_TEAM",
        }
    )
    ids = {c["chunk_id"] for c in result["bm25_candidates"]}
    assert ids == {"hr1", "fin1"}  # 도메인 무관 회수
    assert "secret" not in ids  # ★ 보안: 타 부서 DEPT_ONLY는 항상 배제


def test_bm25_도메인_지정이면_해당_도메인만(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm25_module, "bm25_search", lambda q, top_k: list(_BM25_RESULTS))
    result = bm25_module.bm25_retrieve(
        {
            "question": "경비",
            "requested_domain": "FINANCE_LEGAL",
            "user_department": "HR_TEAM",
        }
    )
    assert [c["chunk_id"] for c in result["bm25_candidates"]] == ["fin1"]


class _FakeClient:
    def __init__(self) -> None:
        self.filters: list[str] = []

    def search(self, name, data, filter, limit, output_fields):  # noqa: A002
        self.filters.append(filter)
        return [
            [
                {
                    "chunk_id": "c1",
                    "distance": 0.9,
                    "entity": {field: "값" for field in output_fields},
                }
            ]
        ]


@pytest.fixture()
def fake_dense(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    fake = _FakeClient()
    monkeypatch.setattr(dense_module, "_embed_query", lambda q: [0.0] * 1024)
    monkeypatch.setattr(dense_module, "get_client", lambda: fake)
    monkeypatch.setattr(dense_module, "get_collection", lambda: "company_docs")
    return fake


def test_dense_도메인_미지정이면_필터에_도메인_절이_없다(fake_dense: _FakeClient) -> None:
    dense_module.dense_retrieve(
        {
            "question": "국가 계약",
            "domain": "FINANCE_LEGAL",  # 라우터 분류 — 무시되어야 함
            "requested_domain": "",
            "user_department": "HR_TEAM",
        }
    )
    assert len(fake_dense.filters) == 1
    assert "domain ==" not in fake_dense.filters[0]
    # ★ 보안 필터는 항상 존재
    assert 'visibility == "ALL"' in fake_dense.filters[0]
    assert 'owning_department == "HR_TEAM"' in fake_dense.filters[0]


def test_dense_도메인_지정이면_필터에_반영된다(fake_dense: _FakeClient) -> None:
    dense_module.dense_retrieve(
        {"question": "경비", "requested_domain": "FINANCE_LEGAL", "user_department": "HR_TEAM"}
    )
    assert 'domain == "FINANCE_LEGAL"' in fake_dense.filters[0]

"""문서 갱신(재적재) 통합 테스트 (임베딩 서버 + Milvus Lite 필요, 기본 skip).

DoD: 문서 1개를 수정 재적재해도 다른 문서 검색 결과가 오염되지 않는다.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_재적재_시_기존_청크가_교체되고_타_문서는_보존된다() -> None:
    from ax_rag.indexer_graph.graph import graph
    from ax_rag.shared import parent_store, vectorstore

    doc = {
        "text": "재택근무는 주 2일까지 허용된다.",
        "source_doc": "재택규정.txt",
        "domain": "HR",
        "owning_department": "HR_TEAM",
        "visibility": "ALL",
        "sections": None,
    }
    graph.invoke(doc)

    rows_before = vectorstore.fetch_all_children(["chunk_id", "source_doc"])
    other_docs_before = {r["chunk_id"] for r in rows_before if r["source_doc"] != "재택규정.txt"}

    # 수정 재적재 (reindex_document.py와 동일 순서)
    vectorstore.delete_by_source_doc("재택규정.txt")
    parent_store.delete_by_source_doc("재택규정.txt")
    graph.invoke({**doc, "text": "재택근무는 주 3일까지 허용된다."})

    rows_after = vectorstore.fetch_all_children(["chunk_id", "source_doc", "text"])
    other_docs_after = {r["chunk_id"] for r in rows_after if r["source_doc"] != "재택규정.txt"}

    # 타 문서 청크는 그대로 (오염 없음)
    assert other_docs_before == other_docs_after
    # 재적재 문서는 새 내용만 존재
    mine = [r["text"] for r in rows_after if r["source_doc"] == "재택규정.txt"]
    assert mine
    assert all("주 3일" in t for t in mine)
    assert all("주 2일" not in t for t in mine)

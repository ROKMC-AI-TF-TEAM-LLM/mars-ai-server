"""indexer_graph E2E 통합 테스트 (임베딩 서버 + Milvus Lite 필요, 기본 skip).

실행 전제: embedding_server(8001) 기동, Milvus Lite 사용 가능(리눅스/L40).
주의: 이 테스트는 실제 데이터 경로(MILVUS_LITE_PATH 등)를 사용하므로
운영 데이터가 있는 환경에서는 .env를 테스트 경로로 바꿔 실행할 것.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DOC_WITH_SECTIONS = {
    "text": "무시됨 (sections 우선)",
    "source_doc": "휴가규정.md",
    "domain": "HR",
    "owning_department": "HR_TEAM",
    "visibility": "ALL",
    "sections": [
        {
            "title": "연차휴가",
            "text": "연차휴가는 매년 15일이 부여된다. 미사용 연차는 최대 5일까지 이월할 수 있다.",
        },
        {
            "title": "육아휴직",
            "text": (
                "육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있다. "
                "급여는 고용보험에서 지급된다."
            ),
        },
    ],
}

_DOC_WITHOUT_SECTIONS = {
    "text": (
        "법인카드는 부서장 승인 후 발급된다. " "월 사용 한도는 직급별로 상이하며 경리팀이 관리한다."
    ),
    "source_doc": "경비규정.txt",
    "domain": "FINANCE_LEGAL",
    "owning_department": "FIN_TEAM",
    "visibility": "DEPT_ONLY",
    "sections": None,
}


def test_섹션_문서와_통짜_문서_적재_E2E() -> None:
    from ax_rag.indexer_graph.graph import graph
    from ax_rag.shared import parent_store, vectorstore
    from ax_rag.shared.config import get_config

    result_1 = graph.invoke(_DOC_WITH_SECTIONS)
    result_2 = graph.invoke(_DOC_WITHOUT_SECTIONS)
    assert result_1["chunks_indexed"] > 0
    assert result_2["chunks_indexed"] > 0

    # company_docs: 두 문서의 자식 청크가 조회된다
    rows = vectorstore.fetch_all_children(["chunk_id", "source_doc", "parent_id"])
    source_docs = {row["source_doc"] for row in rows}
    assert {"휴가규정.md", "경비규정.txt"} <= source_docs

    # document_parents: 자식의 parent_id로 부모 텍스트가 조회된다
    sample = next(row for row in rows if row["source_doc"] == "휴가규정.md")
    parent_text = parent_store.get_parent(sample["parent_id"])
    assert parent_text != ""

    # BM25 인덱스 파일 생성
    config = get_config()
    assert (Path(config.BM25_INDEX_PATH) / "corpus.jsonl").is_file()


def test_BM25_검색이_적재_문서를_찾는다() -> None:
    from ax_rag.shared.bm25_store import bm25_search

    results = bm25_search("육아휴직 기간", top_k=5)
    assert results
    assert any(r["source_doc"] == "휴가규정.md" for r in results)

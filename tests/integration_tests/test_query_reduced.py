"""3단계 축소 그래프 통합 테스트 (임베딩·리랭커 서버 + Milvus Lite 필요, 기본 skip).

실행 전제: test_indexer.py로 샘플 문서가 적재된 상태.
DoD: 샘플 질의 상위 5개가 육안으로 관련 있음 + dense 단독 폴백 동작.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_축소_그래프_E2E_상위_5개_반환() -> None:
    from ax_rag.query_graph.graph import graph

    result = graph.invoke(
        {"question": "육아휴직은 얼마나 쓸 수 있어?", "user_department": "HR_TEAM"}
    )

    assert result["rewritten_query"]  # 라우터가 채운다 (실패 시 원본 폴백)
    chunks = result["retrieved_chunks"]
    assert 1 <= len(chunks) <= 5
    assert all(c["text"] and c["source_doc"] for c in chunks)
    # 육아휴직 관련 문서가 상위에 온다
    assert any("육아" in c["text"] for c in chunks)


def test_타_부서_사용자는_DEPT_ONLY_문서를_받지_못한다() -> None:
    """보안 E2E: 경비규정.txt는 FIN_TEAM DEPT_ONLY로 적재됨."""
    from ax_rag.query_graph.graph import graph

    result = graph.invoke({"question": "법인카드 한도 알려줘", "user_department": "HR_TEAM"})
    sources = {c["source_doc"] for c in result["retrieved_chunks"]}
    assert "경비규정.txt" not in sources

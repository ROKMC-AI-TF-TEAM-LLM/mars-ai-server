"""4단계 전체 그래프 통합 테스트 (vLLM + 임베딩 + 리랭커 + Milvus 필요, 기본 skip).

실행 전제: test_indexer.py로 샘플 문서 적재 + 4개 서비스 전부 기동 (L40).
주의: tool-calling은 개발 노트북(llama.cpp)과 L40(vLLM)의 파서 동작이 다를 수
있으므로 L40 재검증 전까지는 잠정 통과로 취급한다 (roadmap.md).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_멀티턴_맥락_해소_E2E() -> None:
    """roadmap 4단계 DoD: "육아휴직 알려줘" → "그거 얼마나 쓸 수 있어?"."""
    from ax_rag.query_graph.graph import graph

    result = graph.invoke(
        {
            "question": "그거 얼마나 쓸 수 있어?",
            "user_department": "HR_TEAM",
            "conversation_history": [
                {"role": "user", "content": "육아휴직에 대해 알려줘"},
                {"role": "assistant", "content": "육아휴직은 자녀 양육을 위한 휴직 제도입니다."},
            ],
        }
    )

    # rewritten_query가 맥락을 해소했는지 (대명사가 구체화됨)
    assert "그거" not in result["rewritten_query"]
    assert "육아" in result["rewritten_query"]
    assert result["final_answer"]  # finalize 또는 fallback 어느 쪽이든 확정 답변 존재


def test_근거_없는_질문은_fallback에_도달한다() -> None:
    """사내 문서에 없는 주제 → verify fail-closed → fallback."""
    from ax_rag.query_graph.graph import graph
    from ax_rag.query_graph.prompts import FALLBACK_ANSWER

    result = graph.invoke(
        {"question": "우주 정거장 도킹 절차를 알려줘", "user_department": "HR_TEAM"}
    )
    # 관련 근거가 없으므로 생성이 비거나 검증이 걸러서 fallback으로 가야 한다
    assert result["final_answer"] == FALLBACK_ANSWER or result["grounded"] is False

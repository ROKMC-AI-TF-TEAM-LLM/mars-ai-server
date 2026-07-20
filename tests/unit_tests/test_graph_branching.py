"""query_graph 조건부 분기 유닛 테스트 (finalize / increment_retry / fallback)."""

from __future__ import annotations

from ax_rag.query_graph.graph import (
    after_route,
    after_verify,
    fallback,
    finalize,
    increment_retry,
)
from ax_rag.query_graph.prompts import FALLBACK_ANSWER


def test_등록된_도구_intent면_해당_노드로_직행한다() -> None:
    assert after_route({"intent": "SMALLTALK"}) == "SMALLTALK"


def test_DOC_SEARCH나_미설정이면_검색_경로로_간다() -> None:
    for intent in ("DOC_SEARCH", None, ""):
        assert after_route({"intent": intent}) == "dense_retrieve"


def test_레지스트리에_없는_intent는_검색_경로로_간다() -> None:
    assert after_route({"intent": "UNKNOWN_TOOL"}) == "dense_retrieve"


def test_검증_통과면_finalize() -> None:
    assert after_verify({"grounded": True, "retry_count": 0}) == "finalize"


def test_실패_후_재시도_여유가_있으면_increment_retry() -> None:
    # MAX_VERIFY_RETRY=1: 첫 실패(retry_count=0)는 재시도
    assert after_verify({"grounded": False, "retry_count": 0}) == "increment_retry"


def test_재시도_소진이면_fallback() -> None:
    assert after_verify({"grounded": False, "retry_count": 1}) == "fallback"


def test_finalize는_초안을_확정한다() -> None:
    result = finalize({"draft_answer": "확정 답변"})
    assert result["final_answer"] == "확정 답변"
    assert result["pending_intents"] == []  # DOC_SEARCH는 큐에서 소비됨


def test_increment_retry는_횟수를_올린다() -> None:
    assert increment_retry({"retry_count": 0}) == {"retry_count": 1}
    assert increment_retry({}) == {"retry_count": 1}


def test_fallback은_안전한_대체_답변을_확정한다() -> None:
    result = fallback({"verify_reason": "근거 부족"})
    assert result == {"final_answer": FALLBACK_ANSWER}

"""query_graph 조건부 분기 유닛 테스트
(finalize / knowledge_answer / increment_retry / fallback)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ax_rag.query_graph import graph as graph_module
from ax_rag.query_graph.graph import (
    after_route,
    after_verify,
    fallback,
    finalize,
    increment_retry,
    knowledge_answer,
)
from ax_rag.query_graph.prompts import FALLBACK_ANSWER, FALLBACK_DOMAIN_SCOPED_TEMPLATE
from ax_rag.shared.config import get_config

# 검증에 실패했지만 근거는 있는 상태 — 지식 답변 경로로 새면 안 되는 조건.
# (근거가 있는데 검증이 떨어졌다는 건 모델이 수치를 지어냈다는 신호다)
_WITH_CHUNKS: list[dict] = [{"text": "연차는 15일이다", "source_doc": "휴가규정.md"}]


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
    # MAX_VERIFY_RETRY=1: 첫 실패(retry_count=0)는 재시도.
    # 근거가 있어야 재생성에 의미가 있다 (근거 0건이면 지식 답변 경로로 간다)
    assert (
        after_verify({"grounded": False, "retry_count": 0, "retrieved_chunks": _WITH_CHUNKS})
        == "increment_retry"
    )


def test_재시도_소진이면_fallback() -> None:
    assert (
        after_verify({"grounded": False, "retry_count": 1, "retrieved_chunks": _WITH_CHUNKS})
        == "fallback"
    )


def test_finalize는_초안을_확정한다() -> None:
    result = finalize({"draft_answer": "확정 답변"})
    assert result["final_answer"] == "확정 답변"
    assert result["pending_intents"] == []  # DOC_SEARCH는 큐에서 소비됨


def test_increment_retry는_횟수를_올리고_반려_사유를_싣는다() -> None:
    """사유를 넘겨야 재생성이 같은 실수를 반복하지 않는다 (generate가 읽는다)."""
    assert increment_retry({"retry_count": 0, "verify_reason": "근거에 없는 수치: 20"}) == {
        "retry_count": 1,
        "retry_hint": "근거에 없는 수치: 20",
    }
    assert increment_retry({}) == {"retry_count": 1, "retry_hint": ""}


def test_fallback은_안전한_대체_답변을_확정한다() -> None:
    result = fallback({"verify_reason": "근거 부족"})
    assert result == {"final_answer": FALLBACK_ANSWER, "answer_mode": "fallback"}


def test_finalize와_fallback은_답변_경로를_표시한다() -> None:
    """grounded 불리언만으로는 '검증 실패'와 '근거 없이 지식으로 답함'이
    구분되지 않는다 — 감사 로그가 answer_mode로 구분한다."""
    assert finalize({"draft_answer": "확정"})["answer_mode"] == "grounded"
    assert fallback({})["answer_mode"] == "fallback"


# ---------- 지식 기반 답변 분기 (검증을 거치지 않는 경로) ----------


def test_근거_0건이고_도메인_미지정이면_지식_답변으로_간다() -> None:
    """검색이 아무것도 못 찾았으면 재생성해도 결과가 같다 (빈 초안 반복) —
    재시도를 건너뛰고 지식 답변으로 보낸다."""
    assert (
        after_verify({"grounded": False, "retry_count": 0, "retrieved_chunks": []})
        == "knowledge_answer"
    )


def test_근거가_있는데_검증에_실패하면_지식_답변으로_가지_않는다() -> None:
    """⚠️ 회귀 방지선. 근거가 있는데 떨어졌다는 건 모델이 지어냈다는 신호이므로,
    검증 없는 경로로 내보내면 지어낸 내용을 그대로 확정하게 된다 (fail-closed)."""
    for retry_count in (0, 1):
        assert (
            after_verify(
                {"grounded": False, "retry_count": retry_count, "retrieved_chunks": _WITH_CHUNKS}
            )
            != "knowledge_answer"
        )


def test_도메인을_한정했으면_지식_답변으로_가지_않는다() -> None:
    """범위 한정 검색의 실패는 '문서에 없음'이 아니라 '이 범위에 없음'이다 —
    범위를 넓히라고 안내하는 쪽이 옳다."""
    state = {
        "grounded": False,
        "retry_count": 1,
        "retrieved_chunks": [],
        "requested_domain": "DIRECTIVE",
    }
    assert after_verify(state) == "fallback"
    assert "훈령" in fallback(state)["final_answer"]
    assert (
        FALLBACK_DOMAIN_SCOPED_TEMPLATE.format(domain_label="훈령")
        == fallback(state)["final_answer"]
    )


def test_설정_스위치를_끄면_지식_답변을_쓰지_않는다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KNOWLEDGE_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_config.cache_clear()
    try:
        assert (
            after_verify({"grounded": False, "retry_count": 1, "retrieved_chunks": []})
            == "fallback"
        )
    finally:
        get_config.cache_clear()


def test_지식_답변은_경로를_knowledge로_표시하고_도구_답변과_합성한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module, "generate_knowledge_answer", lambda state: "일반 지식 답변")
    result = knowledge_answer(
        {
            "question": "군인 연금은 어떻게 되나요?",
            "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
            "tool_answers": [{"intent": "DISCHARGE_DAYS", "answer": "전역까지 100일"}],
        }
    )
    assert result["answer_mode"] == "knowledge"
    # 계획 순서대로 도구 답변 → 지식 답변
    assert result["final_answer"] == "전역까지 100일\n\n일반 지식 답변"


def test_지식_답변_생성에_실패하면_정형_안내로_되돌린다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """빈 답변이 경고 이벤트만 달고 나가지 않게 한다 — answer_mode가
    fallback이므로 notice도 붙지 않는다."""
    monkeypatch.setattr(graph_module, "generate_knowledge_answer", lambda state: "")
    result = knowledge_answer({"question": "무엇이든", "retrieved_chunks": []})
    assert result == {"final_answer": FALLBACK_ANSWER, "answer_mode": "fallback"}

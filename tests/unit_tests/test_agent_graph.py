"""ReAct 그래프 분기와 추론 스트리밍 유닛 테스트.

SSE 계약이 깨지지 않았다는 증거를 여기서 고정한다: stage 허용값은 그대로이고
thought·step은 **선택 필드로 추가**될 뿐이라, 미들웨어가 무시해도 현행과
동일하게 동작한다 (docs/react_migration_plan.md §5.2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ax_rag.query_graph.agent_tools import ACTION_FINISH, ACTION_SEARCH
from ax_rag.query_graph.graph import (
    after_act,
    after_agent,
    after_verify_agent,
)
from ax_rag.query_graph.stages import NODE_AGENT, status_after_node, status_event_payload
from ax_rag.shared.config import get_config

# SSE 계약이 허용하는 stage 값 (interfaces.md §5). 늘어나면 프론트가 깨진다
_ALLOWED_STAGES = {"route", "tool", "retrieve", "rerank", "generate", "verify"}


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **values: str) -> None:
    """ReAct 분기를 검증하므로 배포 스위치(AGENT_MODE)와 무관하게 켜고 본다."""
    monkeypatch.setenv("AGENT_MODE", "true")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()


# ---------- 루프 분기 ----------


def test_행동을_고르면_act로_간다() -> None:
    assert after_agent({"next_action": ACTION_SEARCH}) == "act"
    assert after_agent({"next_action": "discharge_days"}) == "act"


def test_근거가_있으면_종료시_generate로_간다() -> None:
    state = {"next_action": ACTION_FINISH, "retrieved_chunks": [{"text": "본문"}]}
    assert after_agent(state) == "generate"


def test_검색했지만_근거가_0건이어도_generate로_간다() -> None:
    """빈 초안 → verify fail-closed → knowledge_answer/fallback 경로를 유지해야
    한다. 여기서 다른 곳으로 새면 '근거 0건' 판정이 통째로 사라진다."""
    state = {"next_action": ACTION_FINISH, "retrieved_chunks": [], "search_calls": 1}
    assert after_agent(state) == "generate"


def test_도구_답변만_있으면_finalize로_간다() -> None:
    """검색을 안 한 도구 단독 요청(D-day 등)은 생성·검증을 타지 않는다."""
    state = {"next_action": ACTION_FINISH, "tool_answers": [{"intent": "DISCHARGE_DAYS"}]}
    assert after_agent(state) == "finalize"


def test_파일_예약만_있어도_finalize로_간다() -> None:
    """'이 답변 저장해줘' 단독 요청 — 잡담 응답이 끼어들면 안 된다."""
    state = {"next_action": ACTION_FINISH, "deferred_actions": ["export_hwp"]}
    assert after_agent(state) == "finalize"


def test_아무것도_없으면_직접_응답으로_간다() -> None:
    """인사·잡담·자기소개 경로 (검증 없이 응답, sources 비움)."""
    assert after_agent({"next_action": ACTION_FINISH}) == "direct_answer"


def test_act_후에는_다시_판단하러_간다() -> None:
    assert after_act({"agent_steps": 1}) == "agent"


def test_결정적_단독_행동은_실행_후_되묻지_않는다() -> None:
    """LLM 호출을 한 번 아낀다 — 답이 정해져 있는데 물어볼 이유가 없다."""
    state = {"force_finish": True, "tool_answers": [{"intent": "DISCHARGE_DAYS"}]}
    assert after_act(state) == "finalize"


def test_라운드_상한을_다_쓰면_되묻지_않고_끝낸다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path, MAX_AGENT_STEPS="2")
    try:
        state = {"agent_steps": 2, "retrieved_chunks": [{"text": "본문"}]}
        assert after_act(state) == "generate"
    finally:
        get_config.cache_clear()


# ---------- verify 분기 ----------


def test_검증_통과는_finalize다() -> None:
    assert after_verify_agent({"grounded": True}) == "finalize"


def test_반려되면_재검색을_먼저_시도한다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """★ 문서로 답할 기회를 다 쓰기 전에 검증 없는 지식 답변으로 내려가지 않는다."""
    _env(monkeypatch, tmp_path)
    try:
        state = {"grounded": False, "retrieved_chunks": [], "search_calls": 1, "agent_steps": 1}
        assert after_verify_agent(state) == "retry_search"
    finally:
        get_config.cache_clear()


def test_재검색은_한_번만_한다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env(monkeypatch, tmp_path)
    try:
        state = {
            "grounded": False,
            "retrieved_chunks": [],
            "search_calls": 1,
            "agent_steps": 1,
            "verify_feedback_used": True,  # 이미 한 번 되돌렸다
        }
        # 근거 0건 + 전체 검색이므로 지식 답변 경로로 내려간다
        assert after_verify_agent(state) == "knowledge_answer"
    finally:
        get_config.cache_clear()


def test_되먹임_스위치를_끄면_바로_기존_경로를_탄다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path, AGENT_VERIFY_FEEDBACK="false")
    try:
        state = {"grounded": False, "retrieved_chunks": [{"text": "본문"}], "retry_count": 0}
        assert after_verify_agent(state) == "increment_retry"
    finally:
        get_config.cache_clear()


def test_검색_상한을_다_썼으면_재검색으로_되돌리지_않는다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """되돌려도 에이전트가 검색을 못 하므로 라운드만 낭비된다."""
    _env(monkeypatch, tmp_path, MAX_SEARCH_CALLS="1")
    try:
        state = {
            "grounded": False,
            "retrieved_chunks": [{"text": "본문"}],
            "search_calls": 1,
            "retry_count": 0,
        }
        assert after_verify_agent(state) == "increment_retry"
    finally:
        get_config.cache_clear()


def test_재시도까지_소진하면_fallback이다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env(monkeypatch, tmp_path, MAX_SEARCH_CALLS="1")
    try:
        state = {
            "grounded": False,
            "retrieved_chunks": [{"text": "본문"}],
            "search_calls": 1,
            "retry_count": 1,
        }
        assert after_verify_agent(state) == "fallback"
    finally:
        get_config.cache_clear()


# ---------- 추론 스트리밍 ----------


def test_agent_노드는_다음_행동_기준으로_안내한다() -> None:
    assert status_after_node(NODE_AGENT, {"next_action": ACTION_SEARCH}) == (
        "retrieve",
        "군 내부 문서를 검색하는 중...",
    )
    assert status_after_node(NODE_AGENT, {"next_action": ACTION_FINISH})[0] == "generate"
    assert status_after_node(NODE_AGENT, {"next_action": "discharge_days"}) == (
        "tool",
        "전역일을 계산하는 중...",
    )


def test_재검색은_문구로_구분하되_stage는_그대로다() -> None:
    """계약(stage 허용값)을 늘리지 않고 사용자에게는 다르게 보이게 한다."""
    first = status_after_node(NODE_AGENT, {"next_action": ACTION_SEARCH, "search_calls": 0})
    again = status_after_node(NODE_AGENT, {"next_action": ACTION_SEARCH, "search_calls": 1})
    assert first[0] == again[0] == "retrieve"
    assert first[1] != again[1]


def test_모든_안내의_stage는_계약_허용값_안에_있다() -> None:
    """★ 미들웨어·프론트 계약 불변의 증거."""
    states = [
        {"next_action": ACTION_SEARCH},
        {"next_action": ACTION_SEARCH, "search_calls": 1},
        {"next_action": ACTION_FINISH},
        {"next_action": "discharge_days"},
        {"next_action": "export_hwp"},
    ]
    for state in states:
        stage, _ = status_after_node(NODE_AGENT, state)
        assert stage in _ALLOWED_STAGES
    for node in ("retry_search", "generate", "increment_retry", "fuse", "rerank"):
        status = status_after_node(node, {})
        if status is not None:
            assert status[0] in _ALLOWED_STAGES


def test_thought는_status_이벤트에_선택_필드로_붙는다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path)
    try:
        payload = status_event_payload(
            NODE_AGENT,
            {"next_action": ACTION_SEARCH, "agent_thought": "신청 절차를 찾는다", "agent_steps": 2},
        )
        # 기존 필드는 그대로 — 미들웨어가 thought를 무시해도 동작이 같다
        assert payload["type"] == "status"
        assert payload["stage"] == "retrieve"
        assert payload["message"]
        assert payload["thought"] == "신청 절차를 찾는다"
        assert payload["step"] == 2
    finally:
        get_config.cache_clear()


def test_thought는_agent_노드에서만_붙는다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """이미 지나간 판단이 다른 단계에서 반복 표시되지 않게 한다."""
    _env(monkeypatch, tmp_path)
    try:
        payload = status_event_payload("generate", {"agent_thought": "지난 판단"})
        assert payload is not None
        assert "thought" not in payload
    finally:
        get_config.cache_clear()


def test_스위치를_끄면_thought를_보내지_않는다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """검증을 거치지 않는 텍스트라 즉시 차단할 수단이 있어야 한다."""
    _env(monkeypatch, tmp_path, STREAM_THOUGHTS="false")
    try:
        payload = status_event_payload(
            NODE_AGENT, {"next_action": ACTION_SEARCH, "agent_thought": "판단 근거"}
        )
        assert "thought" not in payload
        assert payload["message"]  # 진행 안내는 그대로 나간다
    finally:
        get_config.cache_clear()

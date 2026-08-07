"""agent 노드(ReAct 판단부) 유닛 테스트.

핵심 계약은 "LLM이 실패해도 현행보다 나빠지지 않는다"이다 —
결정적 폴백·매처 선점·상한 소진 처리를 전부 여기서 고정한다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.agent_tools import ACTION_FINISH, ACTION_SEARCH
from ax_rag.query_graph.nodes import agent as agent_module
from ax_rag.query_graph.nodes.agent import agent
from ax_rag.shared.config import get_config


class _FakeLLM:
    """bind/bind_tools/invoke 계약만 흉내 내는 가짜 LLM."""

    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.captured_messages: list | None = None
        self.call_count = 0

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.call_count += 1
        self.captured_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.response


def _action_response(action: str, thought: str = "이유", query: str = "") -> SimpleNamespace:
    """json_schema 주 경로의 응답 형태 (본문에 JSON)."""
    import json

    payload = {"thought": thought, "action": action, "query": query, "content": ""}
    return SimpleNamespace(tool_calls=[], content=json.dumps(payload, ensure_ascii=False))


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeLLM) -> None:
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)


# ---------- 결정적 경로 (LLM 미사용) ----------


def test_강제_지정된_검색은_LLM_없이_직행하고_바로_종결한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """요청의 tool 필드는 라우터 분류를 이긴다 (엄격 모드 승계)."""
    fake = _FakeLLM(_action_response(ACTION_FINISH))
    _install(monkeypatch, fake)
    result = agent({"question": "육아휴직 얼마나 써?", "intent": "DOC_SEARCH"})
    assert fake.call_count == 0  # LLM 미사용
    assert result["next_action"] == ACTION_SEARCH
    assert result["force_finish"] is True  # 루프를 돌지 않는다


def test_강제_지정된_도구도_LLM_없이_직행한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_action_response(ACTION_FINISH))
    _install(monkeypatch, fake)
    result = agent({"question": "이 답변 저장", "intent": "HWP_EXPORT"})
    assert fake.call_count == 0
    assert result["next_action"] == "export_hwp"


def test_짧은_전역일_질문은_매처만으로_종결한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """현행 라우터의 'LLM 0회' 경로를 그대로 승계한다."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH))
    _install(monkeypatch, fake)
    result = agent({"question": "전역까지 며칠 남았어?"})
    assert fake.call_count == 0
    assert result["next_action"] == "discharge_days"
    assert result["force_finish"] is True


def test_짧은_파일_저장_요청은_예약만_하고_종결한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """지연 도구는 라운드를 쓰지 않는다 — 검색 없이 바로 끝나고 verify 뒤에 실행된다."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH))
    _install(monkeypatch, fake)
    result = agent({"question": "이 답변 한글 파일로 저장해줘"})
    assert fake.call_count == 0
    assert result["next_action"] == ACTION_FINISH
    assert result["deferred_actions"] == ["export_hwp"]


def test_긴_질문은_매처가_걸려도_LLM_판단을_태운다(monkeypatch: pytest.MonkeyPatch) -> None:
    """복합 질문일 수 있다: 매처 도구는 확정하되 루프는 계속 돈다."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH))
    _install(monkeypatch, fake)
    result = agent({"question": "전역까지 며칠 남았는지 알려주고 전역 신청 절차도 자세히 알려줘"})
    assert result["next_action"] == "discharge_days"  # 매처 도구 보장
    assert "force_finish" not in result  # 검색 기회를 남긴다


def test_검색_힌트가_있으면_짧아도_매처_단독으로_끝내지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'찾아서 저장해줘'가 검색 없이 저장만 실행되던 사고의 회귀 방지."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH, query="휴가 규정"))
    _install(monkeypatch, fake)
    result = agent({"question": "휴가 규정 찾아서 한글 파일로 저장해줘"})
    assert fake.call_count == 1  # LLM 판단을 탔다
    assert result["next_action"] == ACTION_SEARCH
    assert result["deferred_actions"] == ["export_hwp"]  # 매처 도구는 예약 보장


# ---------- LLM 판단 ----------


def test_LLM이_검색을_고르면_검색어를_실어_보낸다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_action_response(ACTION_SEARCH, thought="근거를 찾는다", query="육아휴직 기간"))
    _install(monkeypatch, fake)
    result = agent({"question": "육아휴직 얼마나 써?"})
    assert result["next_action"] == ACTION_SEARCH
    assert result["next_action_args"]["query"] == "육아휴직 기간"
    assert result["agent_thought"] == "근거를 찾는다"
    assert result["agent_steps"] == 1


def test_검색어의_파일요청_표현은_결정적으로_제거된다(monkeypatch: pytest.MonkeyPatch) -> None:
    """실측: 오염된 검색어는 리랭크 점수를 30배 무너뜨린다."""
    fake = _FakeLLM(
        _action_response(ACTION_SEARCH, query="해병대 관련 내용을 조사하여 문서로 만들어줘")
    )
    _install(monkeypatch, fake)
    result = agent({"question": "해병대 조사해서 문서로 만들어줘"})
    assert result["next_action_args"]["query"] == "해병대 관련 내용"


def test_intent_이름을_행동으로_보정한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """7B 허용 오차: action="DOC_SEARCH"가 와도 검색을 잃지 않는다."""
    fake = _FakeLLM(_action_response("DOC_SEARCH", query="휴가"))
    _install(monkeypatch, fake)
    result = agent({"question": "휴가 규정 알려줘"})
    assert result["next_action"] == ACTION_SEARCH


def test_미지의_행동은_종료로_처리한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_action_response("전혀_모르는_행동"))
    _install(monkeypatch, fake)
    result = agent({"question": "질문"})
    assert result["next_action"] == ACTION_FINISH


def test_thought가_비면_기본_문구로_채운다(monkeypatch: pytest.MonkeyPatch) -> None:
    """스트림이 끊기지 않게 한다 — 사용자 화면에 빈 칸이 뜨지 않는다."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH, thought="", query="휴가"))
    _install(monkeypatch, fake)
    assert agent({"question": "휴가"})["agent_thought"]


# ---------- 결정적 폴백 ----------


def test_LLM_실패시_1라운드는_원본_질문으로_검색한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 핵심 안전장치: 에이전트가 완전히 실패해도 현행 DOC_SEARCH와 같게 동작한다."""
    _install(monkeypatch, _FakeLLM(exc=RuntimeError("LLM 연결 실패")))
    result = agent({"question": "육아휴직 얼마나 써?"})
    assert result["next_action"] == ACTION_SEARCH
    assert result["next_action_args"]["query"] == "육아휴직 얼마나 써?"


def test_LLM_실패시_2라운드부터는_종료한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """이미 근거를 모았다면 무한히 되묻지 않는다."""
    _install(monkeypatch, _FakeLLM(exc=RuntimeError("LLM 연결 실패")))
    result = agent({"question": "질문", "agent_steps": 1})
    assert result["next_action"] == ACTION_FINISH


def test_응답이_스키마에_안_맞아도_폴백한다(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeLLM(SimpleNamespace(tool_calls=[], content="JSON이 아닌 산문")))
    result = agent({"question": "육아휴직 얼마나 써?"})
    assert result["next_action"] == ACTION_SEARCH


# ---------- 상한 ----------


def test_라운드_상한을_다_쓰면_LLM에_묻지_않고_종료한다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAX_AGENT_STEPS", "2")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_config.cache_clear()
    try:
        fake = _FakeLLM(_action_response(ACTION_SEARCH))
        _install(monkeypatch, fake)
        result = agent({"question": "질문", "agent_steps": 2})
        assert fake.call_count == 0
        assert result["next_action"] == ACTION_FINISH
    finally:
        get_config.cache_clear()


def test_검색_상한을_다_쓰면_검색_요청을_종료로_바꾼다(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LLM이 또 검색하자고 해도 상한이 이긴다 (비용·지연 폭주 차단)."""
    monkeypatch.setenv("MAX_SEARCH_CALLS", "1")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_config.cache_clear()
    try:
        _install(monkeypatch, _FakeLLM(_action_response(ACTION_SEARCH, query="또 검색")))
        result = agent({"question": "질문", "agent_steps": 1, "search_calls": 1})
        assert result["next_action"] == ACTION_FINISH
    finally:
        get_config.cache_clear()


# ---------- 프롬프트 구성 ----------


def test_대화_이력은_대화가_아니라_데이터_블록으로_들어간다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """작은 모델이 '판단' 대신 '대화 이어가기'로 끌려가는 것을 막는다 (라우터와 같은 기법)."""
    fake = _FakeLLM(_action_response(ACTION_SEARCH, query="육아휴직"))
    _install(monkeypatch, fake)
    agent(
        {
            "question": "그거 얼마나 써?",
            "conversation_history": [{"role": "user", "content": "육아휴직 알려줘"}],
        }
    )
    assert fake.captured_messages is not None
    assert len(fake.captured_messages) == 2  # 시스템 + 사용자 블록 하나뿐
    user_text = fake.captured_messages[-1].content
    assert "이어서 답하지 말 것" in user_text
    assert "육아휴직 알려줘" in user_text


def test_스크래치패드가_프롬프트에_실린다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_action_response(ACTION_FINISH))
    _install(monkeypatch, fake)
    agent(
        {
            "question": "질문",
            "agent_steps": 1,
            "agent_scratchpad": [
                {"action": ACTION_SEARCH, "observation": "<observation>\n근거 2건\n</observation>"}
            ],
        }
    )
    user_text = fake.captured_messages[-1].content
    assert "지금까지 한 행동" in user_text
    assert "근거 2건" in user_text

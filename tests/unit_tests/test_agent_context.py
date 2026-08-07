"""에이전트가 **지금 질문**을 처리하는지 검증한다 (멀티턴 회귀 방지).

실측 사고: 6턴짜리 이력이 있는 상태에서 사용자가 "너는 무슨일을 할 수 있어?"라고
새로 물었는데, 에이전트가 직전 턴의 미완 작업("휴가 신청 절차")을 세 라운드
내내 검색했다. thought에도 "관련 부서에 문의하라고 안내한 상태입니다"라고
적혀 있었다 — 지금 질문을 아예 보지 않은 것이다.

원인은 프롬프트 순서였다: 이력이 먼저 오고 질문이 스크래치패드 사이에 묻혀,
작은 모델이 마지막에 읽은 이전 대화를 따라갔다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.agent_tools import ACTION_SEARCH
from ax_rag.query_graph.nodes import act as act_module
from ax_rag.query_graph.nodes import agent as agent_module
from ax_rag.query_graph.nodes.act import act
from ax_rag.query_graph.nodes.agent import AGENT_SYSTEM_PROMPT, agent

_PREVIOUS_TURNS = [
    {"role": "user", "content": "휴가 규정과 휴가 신청 절차에 대해서 알려줘"},
    {"role": "assistant", "content": "신청 절차는 문서에 없습니다. 담당 부서에 문의하세요."},
]


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        return SimpleNamespace(tool_calls=[], content=json.dumps(self.payload, ensure_ascii=False))


# ---------- 프롬프트 배치 (순서가 곧 동작) ----------


def test_지금_질문이_프롬프트_맨_끝에_온다(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 회귀 방지선. 질문이 이력·스크래치패드 뒤에 있어야 모델이 그것을 따라간다."""
    fake = _FakeLLM({"thought": "이유", "action": "finish", "query": "", "content": ""})
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
    agent(
        {
            "question": "너는 무슨일을 할 수 있어?",
            "conversation_history": _PREVIOUS_TURNS,
            "agent_scratchpad": [
                {"action": ACTION_SEARCH, "observation": "<observation>\n근거 0건\n</observation>"}
            ],
            "agent_steps": 1,
        }
    )
    user_text = fake.captured_messages[-1].content
    question_at = user_text.index("너는 무슨일을 할 수 있어?")
    assert question_at > user_text.index("휴가 규정과 휴가 신청 절차")  # 이력보다 뒤
    assert question_at > user_text.index("근거 0건")  # 스크래치패드보다 뒤
    assert "지금 처리할 질문" in user_text


def test_이전_질문을_다시_처리하지_말라고_지시한다() -> None:
    """프롬프트 계약 — 지우면 직전 턴 이어가기 사고가 되살아난다."""
    assert "이전 질문은 이미 끝났으므로 다시 처리하지 않는다" in AGENT_SYSTEM_PROMPT
    assert "검색어는 **지금 질문**에서 뽑는다" in AGENT_SYSTEM_PROMPT
    assert "못 끝낸 이전 작업을 이어서 하지 않는다" in AGENT_SYSTEM_PROMPT


def test_사용자_메시지에도_한_질문만_처리하라고_못박는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM({"thought": "이유", "action": "finish", "query": "", "content": ""})
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
    agent({"question": "질문", "conversation_history": _PREVIOUS_TURNS})
    user_text = fake.captured_messages[-1].content
    assert "위 질문 **하나만** 처리한다" in user_text
    assert "다시 처리하지 않는다" in user_text


# ---------- force_finish 잔류 (LangGraph는 delta를 병합만 한다) ----------


def test_결정이_매번_force_finish를_새로_싣는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """조건부로만 True를 넣으면 이전 라운드 값이 상태에 남아, 새 행동이
    정상 실행됐는데도 after_act가 루프를 조기 종료시킨다."""
    fake = _FakeLLM(
        {"thought": "이유", "action": "search_documents", "query": "새 검색어", "content": ""}
    )
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
    result = agent({"question": "질문", "agent_steps": 1, "force_finish": True})
    assert result["next_action"] == ACTION_SEARCH
    assert result["force_finish"] is False  # 지난 라운드 값이 지워진다


# ---------- 중복 검색어 → 되먹임 차단 ----------


def test_중복_검색어는_새_검색어가_없다는_신호를_남긴다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(act_module, "run_search", lambda state, query: [])
    result = act(
        {
            "question": "질문",
            "next_action": ACTION_SEARCH,
            "next_action_args": {"query": "휴가 신청 절차"},
            "searched_queries": ["휴가 신청 절차"],
        }
    )
    assert result["force_finish"] is True
    assert result["search_ideas_exhausted"] is True


def test_새_검색어가_없으면_반려_되먹임을_걸지_않는다() -> None:
    """되돌려도 같은 검색어가 또 나온다 (실측: 3라운드 내내 동일 검색어)."""
    from ax_rag.query_graph.graph import after_verify

    state = {
        "grounded": False,
        "retrieved_chunks": [],
        "search_calls": 1,
        "agent_steps": 2,
        "search_ideas_exhausted": True,
    }
    assert after_verify(state) != "retry_search"

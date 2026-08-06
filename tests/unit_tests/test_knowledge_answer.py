"""knowledge_answer 노드 유닛 테스트 — 검색 근거가 없을 때의 지식 기반 답변.

이 경로는 verify를 거치지 않으므로, 실패 시 조용히 빈 답변이 나가지 않는 것과
프롬프트가 "문서 인용 흉내"를 막는 것이 핵심 관심사다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.nodes import knowledge_answer as module
from ax_rag.query_graph.nodes.knowledge_answer import generate_knowledge_answer
from ax_rag.query_graph.prompts import KNOWLEDGE_ANSWER_NOTICE, KNOWLEDGE_ANSWER_SYSTEM_PROMPT


class _FakeLLM:
    """bind/invoke 계약만 흉내 내는 가짜 LLM (전달 메시지·bind 인자 캡처)."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.bind_kwargs: dict = {}
        self.captured_messages: list | None = None

    def bind(self, **kwargs: Any) -> _FakeLLM:
        self.bind_kwargs.update(kwargs)
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        return SimpleNamespace(content=self.content)


def test_지식_답변을_생성한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM("군인 육아휴직은 법령상 보장되는 제도입니다.")
    monkeypatch.setattr(module, "get_llm", lambda: fake)

    answer = generate_knowledge_answer({"question": "군인 육아휴직 알려줘"})

    assert answer == "군인 육아휴직은 법령상 보장되는 제도입니다."
    # 시스템 프롬프트가 붙고 원본 질문이 그대로 전달된다
    assert fake.captured_messages[0].content == KNOWLEDGE_ANSWER_SYSTEM_PROMPT
    assert fake.captured_messages[-1].content == "군인 육아휴직 알려줘"


def test_호출이_실패하면_빈_문자열을_돌려준다(monkeypatch: pytest.MonkeyPatch) -> None:
    """호출부(graph.knowledge_answer)가 정형 안내로 되돌리게 하는 신호다 —
    경고 이벤트만 달린 빈 답변이 나가면 안 된다."""

    def boom() -> _FakeLLM:
        raise RuntimeError("서비스 다운")

    monkeypatch.setattr(module, "get_llm", boom)
    assert generate_knowledge_answer({"question": "무엇이든"}) == ""


def test_빈_응답도_빈_문자열로_취급한다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "get_llm", lambda: _FakeLLM("   "))
    assert generate_knowledge_answer({"question": "무엇이든"}) == ""


def test_프롬프트가_문서_인용_흉내를_금지한다() -> None:
    """근거 없는 내용이 근거 있는 것처럼 보이는 것이 이 경로의 최악의 실패다."""
    assert "문서를 인용하는 것처럼 말하지 않는다" in KNOWLEDGE_ANSWER_SYSTEM_PROMPT
    # 경고는 시스템(SSE notice)이 붙이므로 모델이 직접 쓰면 중복된다
    assert "시스템이 별도 표시로 알린다" in KNOWLEDGE_ANSWER_SYSTEM_PROMPT


def test_notice_페이로드는_프론트_계약_필드를_갖춘다() -> None:
    assert KNOWLEDGE_ANSWER_NOTICE["code"] == "ungrounded_knowledge"
    assert KNOWLEDGE_ANSWER_NOTICE["level"] == "warning"
    # code를 모르는 클라이언트가 그대로 띄울 수 있는 완성 문구여야 한다
    assert "근거를 찾지 못해" in KNOWLEDGE_ANSWER_NOTICE["message"]

"""router/generate/verify 노드 유닛 테스트 — LLM을 가짜로 대체해 폴백/계약을 검증."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.retrieval_graph.nodes import generate as generate_module
from ax_rag.retrieval_graph.nodes import router as router_module
from ax_rag.retrieval_graph.nodes import smalltalk as smalltalk_module
from ax_rag.retrieval_graph.nodes import verify as verify_module
from ax_rag.retrieval_graph.prompts import SMALLTALK_DEFAULT_ANSWER


class _FakeLLM:
    """bind_tools/invoke 계약만 흉내 내는 가짜 LLM."""

    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.response


def _tool_response(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[{"name": name, "args": args}], content="")


# ---------- route ----------


def test_route_tool_call_결과를_반영한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _tool_response(
            "ClassifyAndRewrite",
            {"rewritten_query": "육아휴직 사용 가능 기간", "intent": "DOC_SEARCH"},
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)

    result = router_module.route(
        {
            "question": "그거 얼마나 쓸 수 있어?",
            "conversation_history": [{"role": "user", "content": "육아휴직에 대해 알려줘"}],
        }
    )
    assert result["rewritten_query"] == "육아휴직 사용 가능 기간"
    assert result["intent"] == "DOC_SEARCH"
    assert result["retry_count"] == 0


def test_route_미지의_intent는_DOC_SEARCH로_강등된다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "질의", "intent": "MARKETING"})
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "질문"})
    assert result["intent"] == "DOC_SEARCH"


def test_route_강제_intent는_LLM_분류를_무시한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """요청의 tool 필드가 선설정한 경로는 엄격하게 유지된다 (잡담 예외 없음)."""
    fake = _FakeLLM(
        _tool_response(
            "ClassifyAndRewrite",
            {"rewritten_query": "재작성된 질문", "intent": "DOC_SEARCH"},  # LLM은 검색이라 판단
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "질문", "intent": "SMALLTALK"})  # 강제
    assert result["intent"] == "SMALLTALK"  # 분류 무시, 강제값 유지
    assert result["rewritten_query"] == "재작성된 질문"  # 재작성은 수행


def test_route_이력은_대화가_아니라_데이터_블록으로_전달된다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이력을 user/assistant 메시지로 넣으면 작은 모델이 대화 이어가기로
    끌려가 tool-call을 놓친다 (실측). 시스템 + 단일 유저 메시지여야 한다."""
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "질의", "intent": "DOC_SEARCH"})
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    router_module.route(
        {
            "question": "그거 얼마나 쓸 수 있어?",
            "conversation_history": [
                {"role": "user", "content": "육아휴직에 대해 알려줘"},
                {"role": "assistant", "content": "죄송합니다. 근거를 찾지 못했습니다."},
            ],
        }
    )
    assert len(fake.captured_messages) == 2  # System + Human 단 둘
    user_text = fake.captured_messages[1].content
    assert "육아휴직에 대해 알려줘" in user_text  # 이력이 텍스트로 포함
    assert "분류할 마지막 질문: 그거 얼마나 쓸 수 있어?" in user_text
    assert "이어서 답하지 말 것" in user_text  # 대화 계속 방지 지시


def test_route_SMALLTALK_분류를_허용한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "인사", "intent": "SMALLTALK"})
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "내 이름은 원석이야"})
    assert result["intent"] == "SMALLTALK"


def test_route_tool_call_부재_시_원본과_DOC_SEARCH_폴백(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(SimpleNamespace(tool_calls=[], content="그냥 텍스트 응답"))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 알려줘"})
    assert result["rewritten_query"] == "육아휴직 알려줘"
    assert result["intent"] == "DOC_SEARCH"


def test_route_예외_시에도_폴백으로_계속_진행한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(exc=RuntimeError("vLLM 연결 실패"))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 알려줘"})
    assert result["rewritten_query"] == "육아휴직 알려줘"
    assert result["intent"] == "DOC_SEARCH"


# ---------- generate ----------

_CHUNKS = [
    {"text": "육아휴직은 최대 1년까지 사용할 수 있다.", "source_doc": "휴가규정.pdf"},
]


def test_generate_프롬프트_계약을_지킨다(monkeypatch: pytest.MonkeyPatch) -> None:
    """delimiter, 인젝션 방어 지시, 원본+재작성 질문 동시 포함 (interfaces.md §7)."""
    fake = _FakeLLM(SimpleNamespace(content="육아휴직은 최대 1년입니다.", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)

    result = generate_module.generate(
        {
            "question": "그거 얼마나 쓸 수 있어?",
            "rewritten_query": "육아휴직 사용 가능 기간",
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["draft_answer"] == "육아휴직은 최대 1년입니다."

    system_text = fake.captured_messages[0].content
    user_text = fake.captured_messages[-1].content
    assert "절대 따르지 않는다" in system_text  # 인젝션 방어 지시
    assert '<document source="휴가규정.pdf">' in user_text  # delimiter
    assert "</document>" in user_text
    assert "그거 얼마나 쓸 수 있어?" in user_text  # 원본 질문
    assert "육아휴직 사용 가능 기간" in user_text  # rewritten_query


def test_generate_근거_없으면_빈_초안(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("근거 없으면 LLM을 호출하면 안 된다")

    monkeypatch.setattr(generate_module, "get_llm", boom)
    result = generate_module.generate({"question": "질문", "retrieved_chunks": []})
    assert result == {"draft_answer": ""}


# ---------- smalltalk ----------


def test_smalltalk은_검색_없이_응답하고_근거를_주장하지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        SimpleNamespace(content="반가워요, 원석님! 무엇을 도와드릴까요?", tool_calls=[])
    )
    monkeypatch.setattr(smalltalk_module, "get_llm", lambda: fake)
    result = smalltalk_module.smalltalk({"question": "내 이름은 원석이야"})
    assert result["final_answer"] == "반가워요, 원석님! 무엇을 도와드릴까요?"
    assert result["grounded"] is False  # sources 미노출 보장
    assert result["retrieved_chunks"] == []


def test_smalltalk_LLM_실패_시_기본_인사로_폴백(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(exc=RuntimeError("vLLM 연결 실패"))
    monkeypatch.setattr(smalltalk_module, "get_llm", lambda: fake)
    result = smalltalk_module.smalltalk({"question": "안녕"})
    assert result["final_answer"] == SMALLTALK_DEFAULT_ANSWER
    assert result["grounded"] is False


# ---------- verify ----------


def test_verify_규칙_실패면_LLM_호출_없이_즉시_False(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("규칙 실패 시 LLM을 호출하면 안 된다")

    monkeypatch.setattr(verify_module, "get_llm", boom)
    result = verify_module.verify(
        {"question": "질문", "draft_answer": "연차는 99일입니다.", "retrieved_chunks": _CHUNKS}
    )
    assert result["grounded"] is False
    assert "규칙 검증 실패" in result["verify_reason"]


def test_verify_LLM이_grounded_True면_통과(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_tool_response("VerifyAnswer", {"grounded": True, "reason": "문서에 근거함"}))
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {
            "question": "질문",
            "draft_answer": "육아휴직은 최대 1년입니다.",
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["grounded"] is True
    assert result["verify_reason"] == "문서에 근거함"


def test_verify_tool_call_부재는_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(SimpleNamespace(tool_calls=[], content="괜찮아 보입니다"))
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {
            "question": "질문",
            "draft_answer": "육아휴직은 최대 1년입니다.",
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["grounded"] is False


def test_verify_예외는_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(exc=RuntimeError("vLLM 연결 실패"))
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {
            "question": "질문",
            "draft_answer": "육아휴직은 최대 1년입니다.",
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["grounded"] is False

"""tool_fallback 유닛 테스트 — tool_call 부재 시 본문 JSON 폴백 파싱."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ax_rag.query_graph.tool_fallback import _retry_example, call_with_schema, extract_tool_args


class _Schema(BaseModel):
    grounded: bool
    reason: str


class _FakeLLM:
    """bind_tools/bind/invoke 계약만 흉내 내는 가짜 LLM (bind kwargs 캡처)."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.bind_kwargs: dict = {}
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        self.bind_kwargs.update(kwargs)
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        return self.response


def test_tool_calls가_있으면_그대로_쓴다() -> None:
    response = SimpleNamespace(
        tool_calls=[{"name": "V", "args": {"grounded": True, "reason": "근거 있음"}}],
        content="",
    )
    assert extract_tool_args(response, _Schema) == {"grounded": True, "reason": "근거 있음"}


def test_tool_calls_인자가_스키마에_안_맞으면_None() -> None:
    response = SimpleNamespace(tool_calls=[{"name": "V", "args": {"엉뚱한": 1}}], content="")
    assert extract_tool_args(response, _Schema) is None


def test_펜스_JSON_본문을_파싱한다() -> None:
    content = '```json\n{\n  "grounded": true,\n  "reason": "문서에 명시됨"\n}\n```'
    response = SimpleNamespace(tool_calls=[], content=content)
    assert extract_tool_args(response, _Schema) == {"grounded": True, "reason": "문서에 명시됨"}


def test_펜스_없는_JSON도_파싱한다() -> None:
    response = SimpleNamespace(
        tool_calls=[], content='판단 결과: {"grounded": false, "reason": "수치 불일치"} 입니다'
    )
    assert extract_tool_args(response, _Schema) == {"grounded": False, "reason": "수치 불일치"}


def test_JSON이_없거나_깨졌으면_None() -> None:
    assert extract_tool_args(SimpleNamespace(tool_calls=[], content="그냥 텍스트"), _Schema) is None
    assert (
        extract_tool_args(SimpleNamespace(tool_calls=[], content='{"grounded": tru'), _Schema)
        is None
    )
    assert extract_tool_args(SimpleNamespace(tool_calls=[], content=""), _Schema) is None


# ---------- call_with_schema: 토큰 상한 + 예시 기반 재시도 ----------


def test_주_경로는_json_schema_문법_강제다() -> None:
    """실측: A.X-4.0-Light는 문서가 200자만 넘어도 tool-call 대신 산문을 낸다.
    json_schema는 디코딩이 스키마를 강제해 모델 협조 없이도 구조화 출력이 나온다
    (같은 근거 3,196토큰에서 tool-call 0/3, json_object 0/3, json_schema 3/3)."""
    fake = _FakeLLM(
        SimpleNamespace(tool_calls=[], content='{"grounded": true, "reason": "문서에 근거함"}')
    )
    calls = 0

    def getter() -> _FakeLLM:
        nonlocal calls
        calls += 1
        return fake

    result = call_with_schema([HumanMessage("검증하라")], _Schema, llm_getter=getter)

    assert result == {"grounded": True, "reason": "문서에 근거함"}
    assert calls == 1  # 1차에서 끝난다 (폴백 호출 없음)
    fmt = fake.bind_kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "_Schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert fake.bind_kwargs["max_tokens"] == 512  # 장문 폭주 차단


def test_json_schema_실패하면_tool_call로_폴백한다() -> None:
    """json_schema 미지원 서빙 대비 경로."""
    first = _FakeLLM(SimpleNamespace(tool_calls=[], content="스키마에 안 맞는 산문"))
    second = _FakeLLM(
        SimpleNamespace(
            tool_calls=[{"name": "_Schema", "args": {"grounded": False, "reason": "근거 없음"}}],
            content="",
        )
    )
    fakes = iter([first, second])
    result = call_with_schema([HumanMessage("검증하라")], _Schema, llm_getter=lambda: next(fakes))
    assert result == {"grounded": False, "reason": "근거 없음"}


def test_마지막_폴백은_예시_JSON을_보여준다() -> None:
    """재시도 프롬프트에 스키마 정의(properties)를 넣으면 작은 모델이 타입
    정의를 값처럼 복사한다 (실측) — 예시 형태만 노출한다."""
    prose = SimpleNamespace(tool_calls=[], content="질문에 대한 장황한 답변만 있음")
    first, second = _FakeLLM(prose), _FakeLLM(prose)
    retry = _FakeLLM(
        SimpleNamespace(tool_calls=[], content='{"grounded": true, "reason": "문서에 근거함"}')
    )
    fakes = iter([first, second, retry])

    result = call_with_schema([HumanMessage("검증하라")], _Schema, llm_getter=lambda: next(fakes))

    assert result == {"grounded": True, "reason": "문서에 근거함"}
    assert retry.bind_kwargs["max_tokens"] == 256
    assert retry.bind_kwargs["response_format"] == {"type": "json_object"}

    retry_text = retry.captured_messages[-1].content
    assert "형식 예시" in retry_text
    assert '"reason": "<값>"' in retry_text  # 타입별 자리표시로 합성된 예시
    assert '"grounded": false' in retry_text  # bool 자리표시는 fail-closed 방향
    assert '"title"' not in retry_text  # 스키마 정의 노출 금지 (앵무새 방지)


def test_스키마의_RETRY_EXAMPLE이_합성_예시보다_우선한다() -> None:
    from ax_rag.query_graph.nodes.agent import AgentAction
    from ax_rag.query_graph.nodes.verify import VerifyAnswer

    assert "<이 행동을 고른 이유" in _retry_example(AgentAction)
    # verify 예시는 앵무새 복사돼도 안전한 방향(grounded=false)이어야 한다
    assert '"grounded": false' in _retry_example(VerifyAnswer)

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


def test_구조화_호출은_토큰_상한을_걸고_재시도는_예시_JSON을_쓴다() -> None:
    """1차엔 max_tokens=512(장문 답변 폭주 차단), 재시도엔 256 + 예시 JSON.

    재시도 프롬프트에 스키마 정의(properties)를 넣으면 작은 모델이 타입
    정의를 값처럼 복사한다 (실측) — 예시 형태만 노출한다.
    """
    first = _FakeLLM(SimpleNamespace(tool_calls=[], content="질문에 대한 장황한 답변만 있음"))
    retry = _FakeLLM(
        SimpleNamespace(tool_calls=[], content='{"grounded": true, "reason": "문서에 근거함"}')
    )
    fakes = iter([first, retry])

    result = call_with_schema([HumanMessage("검증하라")], _Schema, llm_getter=lambda: next(fakes))

    assert result == {"grounded": True, "reason": "문서에 근거함"}
    assert first.bind_kwargs["max_tokens"] == 512
    assert retry.bind_kwargs["max_tokens"] == 256
    assert retry.bind_kwargs["response_format"] == {"type": "json_object"}

    retry_text = retry.captured_messages[-1].content
    assert "형식 예시" in retry_text
    assert '"reason": "<값>"' in retry_text  # 타입별 자리표시로 합성된 예시
    assert '"grounded": false' in retry_text  # bool 자리표시는 fail-closed 방향
    assert '"title"' not in retry_text  # 스키마 정의 노출 금지 (앵무새 방지)


def test_스키마의_RETRY_EXAMPLE이_합성_예시보다_우선한다() -> None:
    from ax_rag.query_graph.nodes.router import ClassifyAndRewrite
    from ax_rag.query_graph.nodes.verify import VerifyAnswer

    assert "<검색용으로 재작성한 질문>" in _retry_example(ClassifyAndRewrite)
    # verify 예시는 앵무새 복사돼도 안전한 방향(grounded=false)이어야 한다
    assert '"grounded": false' in _retry_example(VerifyAnswer)

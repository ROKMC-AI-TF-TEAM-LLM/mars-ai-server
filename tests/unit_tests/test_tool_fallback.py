"""tool_fallback 유닛 테스트 — tool_call 부재 시 본문 JSON 폴백 파싱."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from ax_rag.retrieval_graph.tool_fallback import extract_tool_args


class _Schema(BaseModel):
    grounded: bool
    reason: str


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

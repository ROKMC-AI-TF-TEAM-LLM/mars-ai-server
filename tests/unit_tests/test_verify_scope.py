"""검증기 판정 범위 계약 테스트.

E2E 실측으로 확인된 오탐의 회귀 방지: **근거에 있는 내용만 인용하고 없는
부분은 "문서에서 확인되지 않는다"고 밝힌 정답 초안**이 "질문을 다 못 답했다"는
이유로 반려되던 문제 (prompts.VERIFY_SYSTEM_PROMPT 주석의 실측 기록 참조).

프롬프트 문구 자체는 계약이므로 여기서 고정한다 — 지우면 오탐이 되살아난다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.nodes import verify as verify_module
from ax_rag.query_graph.nodes.verify import VerifyAnswer, verify
from ax_rag.query_graph.prompts import GENERATE_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT
from ax_rag.query_graph.tool_fallback import json_schema_format

# 실측 사례 재현용 픽스처 (로그에 남은 실제 근거·초안)
_CHUNKS = [
    {
        "text": (
            "본인 결혼 시 5일, 자녀 결혼 시 1일의 경조휴가가 부여된다.\n"
            "배우자 출산 시 10일의 휴가가 부여되며 분할 사용이 가능하다.\n"
            "직계가족 사망 시 5일의 휴가가 부여된다."
        ),
        "source_doc": "휴가규정.md",
    }
]
_HONEST_DRAFT = (
    "## 휴가 규정\n"
    "- 본인 결혼 시: 5일\n"
    "- 자녀 결혼 시: 1일\n"
    "- 배우자 출산 시: 10일 (분할 사용 가능)\n"
    "- 직계가족 사망 시: 5일\n\n"
    "휴가 신청 절차에 대한 정보는 문서에 포함되어 있지 않습니다."
)


class _FakeLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        return self.response


# ---------- 프롬프트 계약 ----------


def test_검증_프롬프트는_판정_범위를_답변_내용으로_한정한다() -> None:
    """지우면 '질문을 다 못 답했다'는 이유의 반려가 되살아난다 (실측)."""
    assert "답변에 실제로 적힌 내용" in VERIFY_SYSTEM_PROMPT
    assert "문서에서 확인되지 않는다" in VERIFY_SYSTEM_PROMPT
    assert "grounded=false로 판정하지 않는다" in VERIFY_SYSTEM_PROMPT


def test_검증_프롬프트는_두_섹션_구조로_바뀌지_않았다() -> None:
    """실측 실패 기록: '[false로 판정한다]/[판정하지 않는다]' 두 섹션으로
    재작성했더니 7B가 앞 섹션만 강하게 적용해 정상 초안 3/3을 탈락시켰다.
    평문 목록 한 줄로 유지할 것."""
    assert "[false로 판정한다]" not in VERIFY_SYSTEM_PROMPT
    assert "[false로 판정하지 않는다]" not in VERIFY_SYSTEM_PROMPT


def test_생성_프롬프트는_모른다에_조치_안내를_붙이지_말라고_지시한다() -> None:
    """'담당 부서에 문의하세요'는 문서에 없는 내용이라 검증기에 빌미를 준다
    (실측: 검증기가 정확히 이 문구를 반려 사유로 지목)."""
    assert "담당 부서에 문의하세요" in GENERATE_SYSTEM_PROMPT
    assert "조치 안내를 덧붙이지 않는다" in GENERATE_SYSTEM_PROMPT


# ---------- 스키마 계약 ----------


@pytest.mark.parametrize("schema", [VerifyAnswer])
def test_모든_속성이_required로_나간다(schema: type) -> None:
    """★ 문법 강제 디코딩에서 기본값 있는 필드는 '관용'이 아니라 '면제'가 된다.
    unsupported가 required가 아니어서 부분 수용이 죽어 있었다 (실측)."""
    payload = json_schema_format(schema)["json_schema"]["schema"]
    assert set(payload["required"]) == set(payload["properties"])


def test_unsupported가_required에_포함된다() -> None:
    payload = json_schema_format(VerifyAnswer)["json_schema"]["schema"]
    assert "unsupported" in payload["required"]


def test_파싱은_여전히_관대하다() -> None:
    """폴백 경로(문법 강제 없음)에서 키가 빠져도 검증에 걸리지 않아야 한다."""
    assert VerifyAnswer.model_validate({"grounded": True, "reason": "근거함"}).unsupported == []


# ---------- 동작 ----------


def test_정직한_모름_진술이_섞인_초안은_통과할_수_있다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """근거 인용 + '문서에 없다' 진술만 있는 초안은 반려 대상이 아니다."""
    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[],
            content='{"grounded": true, "reason": "모든 수치가 근거와 일치", "unsupported": []}',
        )
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify(
        {
            "question": "휴가 규정과 신청 절차 알려줘",
            "draft_answer": _HONEST_DRAFT,
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["grounded"] is True
    # 판정 범위 지시가 실제 프롬프트에 실려 나갔는지
    assert "답변에 실제로 적힌 내용" in fake.captured_messages[0].content


def test_지목_없는_반려는_부분_수용을_시도하지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail-closed는 유지한다 — 지목이 없다고 통과시키지 않는다."""
    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[],
            content='{"grounded": false, "reason": "신청 절차가 문서에 없다", "unsupported": []}',
        )
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify(
        {
            "question": "휴가 규정과 신청 절차 알려줘",
            "draft_answer": _HONEST_DRAFT,
            "retrieved_chunks": _CHUNKS,
        }
    )
    assert result["grounded"] is False

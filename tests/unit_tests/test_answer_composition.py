"""최종 답변 합성과 도구 안내문 계약 — 배선과 무관한 공용 동작.

plan-then-execute 시절 test_plan_execution.py에 있던 것 중 **배선이 아니라
합성·안내문 규칙**을 검증하던 테스트를 옮겨 왔다. ReAct에서는 합성 순서의
근거가 "계획 순서"가 아니라 "실행 순서"로 바뀌었을 뿐, 규칙 자체는 같다.

합성은 verify 뒤의 코드 조립만 허용된다 (fail-closed — LLM 가공 금지).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.graph import fallback, finalize
from ax_rag.query_graph.prompts import FALLBACK_ANSWER


class _FakeLLM:
    """bind/invoke 계약만 흉내 내는 가짜 LLM."""

    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        return self.response


_TOOL_ANSWERS = [{"intent": "DISCHARGE_DAYS", "answer": "전역일까지 D-100, 100일 남았습니다."}]


# ---------- finalize / fallback 합성 ----------


def test_finalize는_실행_순서로_합성한다() -> None:
    state = {
        "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
        "tool_answers": _TOOL_ANSWERS,
        "draft_answer": "전역 신청은 인사담당 부서에 합니다.",
    }
    assert finalize(state)["final_answer"] == (
        "전역일까지 D-100, 100일 남았습니다.\n\n전역 신청은 인사담당 부서에 합니다."
    )
    # 실행 순서가 반대면 합성 순서도 반대
    state["intents"] = ["DOC_SEARCH", "DISCHARGE_DAYS"]
    assert finalize(state)["final_answer"] == (
        "전역 신청은 인사담당 부서에 합니다.\n\n전역일까지 D-100, 100일 남았습니다."
    )


def test_finalize_도구만_실행했으면_도구_답변만_확정한다() -> None:
    state = {"intents": ["DISCHARGE_DAYS"], "tool_answers": _TOOL_ANSWERS, "draft_answer": ""}
    assert finalize(state)["final_answer"] == "전역일까지 D-100, 100일 남았습니다."


def test_fallback은_도구_답변을_유지하고_문서_파트만_대체한다() -> None:
    """도구 답변은 결정적 코드 산출물 — 문서 파트 검증 실패와 무관하게 유지."""
    state = {
        "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
        "tool_answers": _TOOL_ANSWERS,
        "verify_reason": "근거 부족",
    }
    assert fallback(state)["final_answer"] == (
        f"전역일까지 D-100, 100일 남았습니다.\n\n{FALLBACK_ANSWER}"
    )


# ---------- fallback: 도메인 한정 검색 안내 ----------


def test_도메인_한정_검색에서_근거_0건이면_범위를_알려준다() -> None:
    """실측: 훈령(DIRECTIVE) 한정 상태로 휴가를 물으면 근거가 0건이 된다
    (휴가규정.md는 HR 적재). "문서가 없다"가 아니라 "이 범위에 없다"로 안내해야
    사용자가 범위를 좁혀둔 걸 알아챈다."""
    answer = fallback({"requested_domain": "DIRECTIVE", "retrieved_chunks": []})["final_answer"]
    assert "훈령" in answer  # DOMAIN_LABELS의 한글 라벨
    assert "전체" in answer  # 조치 안내
    assert answer != FALLBACK_ANSWER


def test_도메인_한정이어도_근거가_있었으면_기본_문구다() -> None:
    """근거는 찾았는데 검증에서 떨어진 경우(지어낸 수치 등)는 범위 탓이 아니다."""
    state = {
        "requested_domain": "DIRECTIVE",
        "retrieved_chunks": [{"text": "본문", "source_doc": "훈령.pdf"}],
    }
    assert fallback(state)["final_answer"] == FALLBACK_ANSWER


def test_도메인_한정이_없으면_기본_문구다() -> None:
    assert fallback({"retrieved_chunks": []})["final_answer"] == FALLBACK_ANSWER
    assert (
        fallback({"requested_domain": "", "retrieved_chunks": []})["final_answer"]
        == FALLBACK_ANSWER
    )


def test_도메인_한정_안내도_도구_답변을_유지한다() -> None:
    state = {
        "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
        "tool_answers": _TOOL_ANSWERS,
        "requested_domain": "MANUAL",
        "retrieved_chunks": [],
    }
    answer = fallback(state)["final_answer"]
    assert answer.startswith("전역일까지 D-100, 100일 남았습니다.\n\n")
    assert "교범" in answer  # MANUAL의 한글 라벨


# ---------- 도구 처리분 안내문 (generate / verify 공용) ----------


def test_TOOL_HANDLED_LABELS는_전_도구를_예시없이_커버한다() -> None:
    """generate/verify 안내문 계약: 라우터용 예시 문구가 안내문에 실리면 7B
    검증기가 답변 내용으로 착각해 grounded=false 오탐 (실측). 라벨은 예시 금지."""
    from ax_rag.query_graph.tools import TOOL_HANDLED_LABELS, TOOL_NODES

    for name in TOOL_NODES:
        assert name in TOOL_HANDLED_LABELS  # 새 도구 등록 시 라벨도 필수
    for label in TOOL_HANDLED_LABELS.values():
        assert "예:" not in label and "만들어줘" not in label and "줘" not in label


def test_generate는_도구가_처리한_요청을_답하지_말라고_안내한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """도구 처리분을 generate가 창작하면 verify가 문서 파트 전체를 탈락시킨다
    (E2E 실측). 유형 설명만 넣고 수치는 넣지 않는다."""
    from ax_rag.query_graph.nodes import generate as generate_module

    fake = _FakeLLM(SimpleNamespace(content="연차 이월은 5일까지 가능합니다.", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)

    generate_module.generate(
        {
            "question": "전역까지 며칠 남았는지랑 연차 이월 규정 알려줘",
            "rewritten_query": "연차 이월 규정",
            "retrieved_chunks": [{"text": "이월은 최대 5일.", "source_doc": "휴가규정.md"}],
            "tool_answers": [{"intent": "DISCHARGE_DAYS", "answer": "D-140, 140일 남았습니다."}],
        }
    )
    user_text = fake.captured_messages[-1].content
    assert "답하지 말고" in user_text  # 중복 답변 방지 안내 존재
    assert "전역" in user_text  # 도구 유형 라벨(TOOL_HANDLED_LABELS) 포함
    assert "D-140" not in user_text and "140일" not in user_text  # 수치는 미포함


def test_generate는_도구_처리분이_없으면_안내를_붙이지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ax_rag.query_graph.nodes import generate as generate_module

    fake = _FakeLLM(SimpleNamespace(content="답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)
    generate_module.generate(
        {
            "question": "연차 이월 규정 알려줘",
            "retrieved_chunks": [{"text": "이월은 최대 5일.", "source_doc": "휴가규정.md"}],
        }
    )
    assert "답하지 말고" not in fake.captured_messages[-1].content


def test_파일_저장_예약도_안내문에_포함된다(monkeypatch: pytest.MonkeyPatch) -> None:
    """지연 도구는 아직 실행 전이지만 안내문에 들어가야 한다 — 빠뜨리면
    verify가 "답변이 파일 저장을 안 다뤘다"고 오탐한다 (실측)."""
    from ax_rag.query_graph.nodes import generate as generate_module

    fake = _FakeLLM(SimpleNamespace(content="답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)
    generate_module.generate(
        {
            "question": "휴가 규정 찾아서 한글 파일로 저장해줘",
            # act가 검색·예약을 실행 순서로 기록한 상태
            "intents": ["DOC_SEARCH", "HWP_EXPORT"],
            "retrieved_chunks": [{"text": "이월은 최대 5일.", "source_doc": "휴가규정.md"}],
        }
    )
    assert "한글(HWPX) 문서 파일로 저장" in fake.captured_messages[-1].content


def test_verify는_도구_처리분을_판정_범위에서_제외하라고_안내한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify가 도구 몫(답변에 없는 부분)을 이유로 문서 파트를 탈락시키는
    오탐 방지 (E2E 실측). 수치는 넣지 않고 유형 설명만 전달한다."""
    from ax_rag.query_graph.nodes import verify as verify_module

    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[],
            content='{"grounded": true, "reason": "문서에 근거함", "unsupported": []}',
        )
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {
            "question": "전역까지 며칠 남았는지랑 연차 이월 규정 알려줘",
            "draft_answer": "이월은 최대 5일까지 가능합니다.",
            "retrieved_chunks": [{"text": "이월은 최대 5일.", "source_doc": "휴가규정.md"}],
            "tool_answers": [{"intent": "DISCHARGE_DAYS", "answer": "D-140, 140일 남았습니다."}],
        }
    )
    assert result["grounded"] is True
    user_text = fake.captured_messages[-1].content
    assert "검증 대상이 아니다" in user_text.replace("\n", " ")
    assert "D-140" not in user_text  # 도구 수치는 미포함

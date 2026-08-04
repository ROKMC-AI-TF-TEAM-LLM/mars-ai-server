"""router/generate/verify 노드 유닛 테스트 — LLM을 가짜로 대체해 폴백/계약을 검증."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.nodes import generate as generate_module
from ax_rag.query_graph.nodes import router as router_module
from ax_rag.query_graph.nodes import smalltalk as smalltalk_module
from ax_rag.query_graph.nodes import verify as verify_module
from ax_rag.query_graph.prompts import SMALLTALK_DEFAULT_ANSWER


class _FakeLLM:
    """bind_tools/invoke 계약만 흉내 내는 가짜 LLM."""

    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.captured_messages: list | None = None

    def bind_tools(self, tools: list, **kwargs: Any) -> _FakeLLM:
        return self

    def bind(self, **kwargs: Any) -> _FakeLLM:
        self.bind_kwargs = kwargs
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
            {"rewritten_query": "육아휴직 사용 가능 기간", "intents": ["DOC_SEARCH"]},
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
    assert result["intents"] == ["DOC_SEARCH"]
    assert result["pending_intents"] == ["DOC_SEARCH"]
    assert result["intent"] == "DOC_SEARCH"  # 대표값 (계획 첫 항목)
    assert result["retry_count"] == 0


def test_route_미지의_intent는_DOC_SEARCH로_강등된다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "질의", "intents": ["MARKETING"]})
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "질문"})
    assert result["intents"] == ["DOC_SEARCH"]
    assert result["intent"] == "DOC_SEARCH"


def test_route_강제_intent는_LLM_분류를_무시한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """요청의 tool 필드가 선설정한 경로는 엄격하게 유지된다 (잡담 예외 없음)."""
    fake = _FakeLLM(
        _tool_response(
            "ClassifyAndRewrite",
            # LLM은 검색이라 판단
            {"rewritten_query": "재작성된 질문", "intents": ["DOC_SEARCH"]},
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "질문", "intent": "SMALLTALK"})  # 강제
    assert result["intent"] == "SMALLTALK"  # 분류 무시, 강제값 유지
    assert result["intents"] == ["SMALLTALK"]  # 계획도 강제 경로 하나로 고정
    assert result["rewritten_query"] == "재작성된 질문"  # 재작성은 수행


def test_route_이력은_대화가_아니라_데이터_블록으로_전달된다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이력을 user/assistant 메시지로 넣으면 작은 모델이 대화 이어가기로
    끌려가 tool-call을 놓친다 (실측). 시스템 + 단일 유저 메시지여야 한다."""
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "질의", "intents": ["DOC_SEARCH"]})
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


def test_route_결정적_매처는_LLM_없이_즉시_라우팅한다(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("짧은 질문을 매처가 잡으면 LLM을 호출하면 안 된다")

    monkeypatch.setattr(router_module, "get_llm", boom)
    result = router_module.route({"question": "내 전역일은 2026년 12월 1일이야"})
    assert result["intent"] == "DISCHARGE_DAYS"
    assert result["intents"] == ["DISCHARGE_DAYS"]


def test_route_SMALLTALK_분류를_허용한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(
        _tool_response("ClassifyAndRewrite", {"rewritten_query": "인사", "intents": ["SMALLTALK"]})
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "내 이름은 원석이야"})
    assert result["intent"] == "SMALLTALK"
    assert result["intents"] == ["SMALLTALK"]  # 단독 잡담은 계획으로 허용


def test_route_tool_call_부재_시_원본과_DOC_SEARCH_폴백(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(SimpleNamespace(tool_calls=[], content="그냥 텍스트 응답"))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 알려줘"})
    assert result["rewritten_query"] == "육아휴직 알려줘"
    assert result["intents"] == ["DOC_SEARCH"]


def test_route_예외_시에도_폴백으로_계속_진행한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(exc=RuntimeError("vLLM 연결 실패"))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 알려줘"})
    assert result["rewritten_query"] == "육아휴직 알려줘"
    assert result["intents"] == ["DOC_SEARCH"]


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


def test_generate는_설정_온도로_호출한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """답변 생성만 GENERATE_TEMPERATURE(기본 0.2) — 라우터·verify는 0 유지."""
    from ax_rag.shared.config import get_config

    fake = _FakeLLM(SimpleNamespace(content="답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)
    generate_module.generate({"question": "질문", "retrieved_chunks": _CHUNKS})
    assert fake.bind_kwargs == {"temperature": get_config().GENERATE_TEMPERATURE}


def test_재생성이면_반려_사유를_프롬프트에_넣는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """실측: 사유를 안 알려주면 재생성이 같은 실수를 반복한다
    (1차 777자 → 2차 768자, 둘 다 '신청 방법이 문서에 없다'로 탈락 후 fallback)."""
    fake = _FakeLLM(SimpleNamespace(content="다시 쓴 답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)

    generate_module.generate(
        {
            "question": "휴가 규정 알려줘",
            "retrieved_chunks": _CHUNKS,
            "retry_count": 1,
            "retry_hint": "신청 방법에 대한 구체적인 정보가 문서에 포함되어 있지 않습니다.",
        }
    )
    user_text = fake.captured_messages[-1].content
    assert "반려" in user_text
    assert "신청 방법에 대한 구체적인 정보가 문서에 포함되어 있지 않습니다." in user_text
    # 과잉 교정 방지 지시가 함께 들어가야 한다 (사유만 주면 답변 전체를 줄여버린다)
    assert "나머지 내용은 줄이지 말고" in user_text


def test_첫_생성에는_재작성_지시가_없다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(SimpleNamespace(content="답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)
    generate_module.generate({"question": "질문", "retrieved_chunks": _CHUNKS})
    assert "반려" not in fake.captured_messages[-1].content


def test_사유가_비어_있으면_재작성_지시를_붙이지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify가 사유 없이 떨어뜨린 경우 빈 지시문으로 프롬프트를 오염시키지 않는다."""
    fake = _FakeLLM(SimpleNamespace(content="답변", tool_calls=[]))
    monkeypatch.setattr(generate_module, "get_llm", lambda: fake)
    generate_module.generate(
        {"question": "질문", "retrieved_chunks": _CHUNKS, "retry_count": 1, "retry_hint": "  "}
    )
    assert "반려" not in fake.captured_messages[-1].content


def test_increment_retry가_사유를_실어_보낸다() -> None:
    from ax_rag.query_graph.graph import increment_retry

    result = increment_retry({"retry_count": 0, "verify_reason": "근거에 없는 수치: 20"})
    assert result["retry_count"] == 1
    assert result["retry_hint"] == "근거에 없는 수치: 20"


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


def test_smalltalk은_온도를_올려_호출한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """잡담은 자연스러움 우선 — temperature 0.7 (검색·검증 경로는 0 유지)."""
    fake = _FakeLLM(SimpleNamespace(content="반가워요!", tool_calls=[]))
    monkeypatch.setattr(smalltalk_module, "get_llm", lambda: fake)
    smalltalk_module.smalltalk({"question": "안녕"})
    assert fake.bind_kwargs == {"temperature": 0.7}


def test_smalltalk_LLM_실패_시_기본_인사로_폴백(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(exc=RuntimeError("vLLM 연결 실패"))
    monkeypatch.setattr(smalltalk_module, "get_llm", lambda: fake)
    result = smalltalk_module.smalltalk({"question": "안녕"})
    assert result["final_answer"] == SMALLTALK_DEFAULT_ANSWER
    assert result["grounded"] is False


# ---------- verify ----------


def test_generate_프롬프트는_풍부한_부분_답변을_요구한다() -> None:
    """모르는 부분이 있다고 답변 전체를 줄이면 안 된다 (verify 개정과 짝)."""
    from ax_rag.query_graph.prompts import GENERATE_SYSTEM_PROMPT

    assert "한두 문장으로 끝내지 않는다" in GENERATE_SYSTEM_PROMPT
    assert "답할 수 있는 부분은 그대로 충실히 답한다" in GENERATE_SYSTEM_PROMPT
    # 근거 준수 지시는 그대로 유지돼야 한다
    assert "지어내 추가하지 않는다" in GENERATE_SYSTEM_PROMPT
    assert "예시를 지어내지 않는다" in GENERATE_SYSTEM_PROMPT


class _SequenceLLM:
    """호출마다 다른 응답을 돌려주는 가짜 LLM (부분 수용의 2회 검증용)."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools: list, **kwargs: object) -> _SequenceLLM:
        return self

    def bind(self, **kwargs: object) -> _SequenceLLM:
        return self

    def invoke(self, messages: list) -> object:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


_PARTIAL_DRAFT = "병가는 연 60일 한도입니다.\n병가 신청은 지휘관의 승인을 받아야 합니다."


def test_verify_부분_수용은_지목된_문장만_빼고_통과시킨다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실측: 한도·진단서는 정확한데 '지휘관 승인'을 지어내 답변 전체가 버려졌다."""
    fake = _SequenceLLM(
        [
            _tool_response(
                "VerifyAnswer",
                {
                    "grounded": False,
                    "reason": "지휘관 승인은 문서에 없다",
                    "unsupported": ["병가 신청은 지휘관의 승인을 받아야 합니다."],
                },
            ),
            _tool_response("VerifyAnswer", {"grounded": True, "reason": "남은 내용은 근거함"}),
        ]
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {"question": "병가 규정", "draft_answer": _PARTIAL_DRAFT, "retrieved_chunks": _CHUNKS}
    )
    assert result["grounded"] is True
    assert "지휘관" not in result["draft_answer"]  # 근거 없는 문장 제거
    assert "연 60일 한도" in result["draft_answer"]  # 근거 있는 내용은 유지
    assert "제외했습니다" in result["draft_answer"]  # 사용자 안내 부착
    assert fake.calls == 2  # 정제본을 반드시 재검증한다


def test_verify_정제본이_재검증에_떨어지면_반려_유지(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ fail-closed: 최종적으로 검증을 통과한 텍스트만 내보낸다."""
    fake = _SequenceLLM(
        [
            _tool_response(
                "VerifyAnswer",
                {
                    "grounded": False,
                    "reason": "근거 없음",
                    "unsupported": ["병가 신청은 지휘관의 승인을 받아야 합니다."],
                },
            ),
            _tool_response("VerifyAnswer", {"grounded": False, "reason": "여전히 근거 없음"}),
        ]
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {"question": "병가 규정", "draft_answer": _PARTIAL_DRAFT, "retrieved_chunks": _CHUNKS}
    )
    assert result["grounded"] is False
    assert "draft_answer" not in result  # 초안을 바꾸지 않는다


def test_verify_지목이_없으면_부분_수용을_시도하지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _SequenceLLM(
        [_tool_response("VerifyAnswer", {"grounded": False, "reason": "근거 없음"})]
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {"question": "병가 규정", "draft_answer": _PARTIAL_DRAFT, "retrieved_chunks": _CHUNKS}
    )
    assert result["grounded"] is False
    assert fake.calls == 1  # 재검증 호출 없음


def test_verify_전제_미충족이면_LLM_호출_없이_즉시_False(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """빈 답변·근거 0건은 LLM에 물을 것이 없다 — 호출을 아끼고 fail-closed."""

    def boom() -> None:
        raise AssertionError("전제 미충족 시 LLM을 호출하면 안 된다")

    monkeypatch.setattr(verify_module, "get_llm", boom)
    for state in (
        {"question": "질문", "draft_answer": "   ", "retrieved_chunks": _CHUNKS},
        {"question": "질문", "draft_answer": "연차는 15일입니다.", "retrieved_chunks": []},
    ):
        result = verify_module.verify(state)
        assert result["grounded"] is False
        assert "전제 미충족" in result["verify_reason"]


def test_verify_지어낸_수치는_LLM에_판정을_맡긴다(monkeypatch: pytest.MonkeyPatch) -> None:
    """규칙으로 미리 막지 않는다. 실측 근거: LLM이 지어낸 일수·이월 한도·기한·
    조항 번호를 3회씩 전부 grounded=False로 잡았다 (15/15)."""
    fake = _FakeLLM(
        _tool_response("VerifyAnswer", {"grounded": False, "reason": "근거는 15일인데 99일이라 함"})
    )
    monkeypatch.setattr(verify_module, "get_llm", lambda: fake)
    result = verify_module.verify(
        {"question": "질문", "draft_answer": "연차는 99일입니다.", "retrieved_chunks": _CHUNKS}
    )
    assert fake.captured_messages is not None  # LLM이 실제로 호출됐다
    assert result["grounded"] is False


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

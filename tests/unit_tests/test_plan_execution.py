"""plan-then-execute 유닛 테스트 — 계획 정규화, 실행 큐, 도구 결과 합성.

복합 질문(예: "전역까지 며칠 남았고 절차는?")이 여러 경로를 순차 실행한 뒤
계획 순서로 합성되는 계약을 검증한다. 합성은 verify 뒤의 코드 조립만
허용된다 (fail-closed — LLM 가공 금지).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import main
import pytest

from ax_rag.query_graph import graph as graph_module
from ax_rag.query_graph.graph import after_route, fallback, finalize, next_step
from ax_rag.query_graph.nodes import router as router_module
from ax_rag.query_graph.nodes.router import _normalize_plan
from ax_rag.query_graph.prompts import FALLBACK_ANSWER
from ax_rag.query_graph.tools import DOC_SEARCH, execution_queue


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


def _classify_response(rewritten: str, intents: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        tool_calls=[
            {
                "name": "ClassifyAndRewrite",
                "args": {"rewritten_query": rewritten, "intents": intents},
            }
        ],
        content="",
    )


# ---------- 계획 정규화 ----------


def test_정규화는_미지값을_제거하고_중복_없이_순서를_유지한다() -> None:
    plan = _normalize_plan(["DOC_SEARCH", "MARKETING", "DISCHARGE_DAYS", "DOC_SEARCH"], [])
    assert plan == ["DOC_SEARCH", "DISCHARGE_DAYS"]


def test_정규화_빈_계획이나_전량_미지값은_DOC_SEARCH_폴백() -> None:
    assert _normalize_plan([], []) == [DOC_SEARCH]
    assert _normalize_plan(["NO_SUCH"], []) == [DOC_SEARCH]


def test_정규화는_매처_확정_도구를_보장_포함한다() -> None:
    """LLM이 빠뜨려도 결정적 매처가 잡은 도구는 계획에 들어간다."""
    assert _normalize_plan(["DOC_SEARCH"], ["DISCHARGE_DAYS"]) == ["DOC_SEARCH", "DISCHARGE_DAYS"]


def test_정규화_SMALLTALK은_단독일_때만_허용된다() -> None:
    """단독 전용 도구는 업무 경로와 합성하지 않는다 (verify 밖 자유 생성)."""
    assert _normalize_plan(["SMALLTALK"], []) == ["SMALLTALK"]
    assert _normalize_plan(["SMALLTALK", "DOC_SEARCH"], []) == ["DOC_SEARCH"]


def test_정규화는_계획_길이를_상한으로_절단한다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "valid_intents", lambda: ("DOC_SEARCH", "T1", "T2", "T3"))
    assert _normalize_plan(["T1", "T2", "T3", "DOC_SEARCH"], []) == ["T1", "T2", "T3"]


# ---------- 실행 큐 ----------


def test_실행_큐는_도구_먼저_DOC_SEARCH는_마지막() -> None:
    assert execution_queue(["DOC_SEARCH", "DISCHARGE_DAYS"]) == ["DISCHARGE_DAYS", "DOC_SEARCH"]
    assert execution_queue(["DISCHARGE_DAYS"]) == ["DISCHARGE_DAYS"]
    assert execution_queue(["DOC_SEARCH"]) == ["DOC_SEARCH"]


# ---------- route: 복합 계획 ----------


def test_route_복합_질문은_계획_여러_개를_반환한다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLLM(_classify_response("휴가 규정", ["DISCHARGE_DAYS", "DOC_SEARCH"]))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "휴가 규정이랑 경비 처리 절차도 알려줘"})
    assert result["intents"] == ["DISCHARGE_DAYS", "DOC_SEARCH"]  # 합성 순서 = 계획 순서
    assert result["pending_intents"] == ["DISCHARGE_DAYS", "DOC_SEARCH"]  # 도구 먼저 실행


def test_route_긴_질문은_매처_히트여도_LLM을_태우고_병합한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """긴 질문은 복합일 수 있다: LLM 분류를 수행하되 매처 도구는 보장 포함."""
    fake = _FakeLLM(_classify_response("전역 신청 절차", ["DOC_SEARCH"]))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route(
        {"question": "전역까지 며칠 남았는지 알려주고 전역 신청 절차도 자세히 알려줘"}
    )
    assert fake.captured_messages is not None  # LLM 호출됨
    assert result["intents"] == ["DOC_SEARCH", "DISCHARGE_DAYS"]  # 매처 도구 병합
    assert result["pending_intents"] == ["DISCHARGE_DAYS", "DOC_SEARCH"]


@pytest.mark.parametrize(
    ("polluted", "expected"),
    [
        ("해병대 관련 내용을 조사하여 문서로 만들어줘", "해병대 관련 내용"),
        ("휴가 규정 찾아서 파일로 만들어줘", "휴가 규정"),
        ("탄약 훈령 요약해서 한글 파일로 저장해줘", "탄약 훈령"),
        ("위 내용을 문서화해줘", "위 내용"),
        ("휴가 제도", "휴가 제도"),  # 파일 표현 없으면 그대로 (명사 "제도" 보존)
        ("문서 검색해줘", "문서 검색해줘"),  # "문서"만으로는 제거 안 함
    ],
)
def test_strip_file_phrases_파일요청_표현을_걷어낸다(polluted: str, expected: str) -> None:
    """실측: 재작성 쿼리에 '문서로 만들어줘'가 남으면 리랭크 점수가 30배
    붕괴(0.738→0.022)해 검색이 전멸한다. 결정적 정리로 검색어만 남긴다."""
    assert router_module._strip_file_phrases(polluted) == expected


def test_strip_file_phrases_전부_제거되면_원본_유지() -> None:
    """검색어가 실종될 정도로 깎이면 오염된 원본이라도 유지한다 (빈 쿼리 방지)."""
    assert router_module._strip_file_phrases("문서로 만들어줘") == "문서로 만들어줘"


def test_route_검색_파일_복합계획이면_쿼리를_정리한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM이 재작성에서 파일 요청 표현을 못 뗀 경우 코드가 정리한다."""
    fake = _FakeLLM(
        _classify_response(
            "해병대 관련 내용을 조사하여 문서로 만들어줘", ["DOC_SEARCH", "HWP_EXPORT"]
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "해병대 관련 내용을 조사하고 문서를 만들어줘"})
    assert result["intents"] == ["DOC_SEARCH", "HWP_EXPORT"]
    assert result["rewritten_query"] == "해병대 관련 내용"  # 오염 제거됨


def test_HWP_EXPORT_설명은_한글없는_문서표현도_포함한다() -> None:
    """라우터 프롬프트 계약: '문서 만들어줘'(한글 없음)도 HWP_EXPORT로 분류되게
    설명이 '한글'에 얽매이지 않아야 한다 (실측: 좁은 설명이라 복합 의도 유실)."""
    from ax_rag.query_graph.nodes.router import ROUTER_SYSTEM_PROMPT
    from ax_rag.query_graph.tools import TOOL_DESCRIPTIONS

    desc = TOOL_DESCRIPTIONS["HWP_EXPORT"]
    assert "문서로 만들어줘" in desc  # 한글 없는 표현 포함
    assert "DOC_SEARCH, HWP_EXPORT" in desc  # 검색+저장 복합 예시
    assert "HWP_EXPORT" in ROUTER_SYSTEM_PROMPT  # 프롬프트에 실제 반영


def test_TOOL_HANDLED_LABELS는_전_도구를_예시없이_커버한다() -> None:
    """generate/verify 안내문 계약: 라우터용 예시 문구가 안내문에 실리면 7B
    검증기가 답변 내용으로 착각해 grounded=false 오탐 (실측: '해병대 조사해서
    문서로 만들어줘' 예시가 실제 질문 주제와 겹칠 때). 라벨은 예시 금지."""
    from ax_rag.query_graph.tools import TOOL_HANDLED_LABELS, TOOL_NODES

    for name in TOOL_NODES:
        assert name in TOOL_HANDLED_LABELS  # 새 도구 등록 시 라벨도 필수
    for label in TOOL_HANDLED_LABELS.values():
        assert "예:" not in label and "만들어줘" not in label and "줘" not in label


@pytest.mark.parametrize(
    "question",
    [
        "규정 찾아서 한글 파일로 저장해줘",
        "해병대에 대해서 조사하고 한글 문서로 만들어줘",  # "조사" — 실측 유실 케이스
        "휴가 규정 정리해서 한글 문서로 저장해줘",
        "탄약 훈령 요약해서 한글 파일로 만들어줘",
    ],
)
def test_route_짧아도_검색_힌트가_있으면_LLM을_병행한다(
    monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    """검색 선행 표현(찾아·조사·정리·요약 등)이 섞인 짧은 질문이 매처 단독으로
    검색 없이 저장만 실행되는 사고 방지 — 힌트가 있으면 LLM 분류를 태운다."""
    fake = _FakeLLM(_classify_response("검색 쿼리", ["DOC_SEARCH"]))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": question})
    assert fake.captured_messages is not None  # LLM 호출됨 (단독 종결 아님)
    assert result["intents"] == ["DOC_SEARCH", "HWP_EXPORT"]  # 매처 도구 병합
    assert result["pending_intents"] == ["DOC_SEARCH", "HWP_EXPORT"]  # 검색 → 후처리 순


def test_route_LLM_실패해도_매처_확정_도구는_잃지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(exc=RuntimeError("LLM 연결 실패"))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route(
        {"question": "전역까지 며칠 남았는지 알려주고 전역 신청 절차도 자세히 알려줘"}
    )
    assert result["intents"] == ["DISCHARGE_DAYS", "DOC_SEARCH"]


def test_route_문자열_intents_응답도_리스트로_보정해_수용한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7B 허용 오차: intents가 리스트가 아니라 문자열로 와도 검증 탈락시키지 않는다."""
    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[
                {
                    "name": "ClassifyAndRewrite",
                    "args": {"rewritten_query": "육아휴직 기간", "intents": "DOC_SEARCH"},
                }
            ],
            content="",
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 얼마나 써?"})
    assert result["intents"] == ["DOC_SEARCH"]
    assert result["rewritten_query"] == "육아휴직 기간"  # 재작성이 유실되지 않는다


def test_route_단수형_intent_응답도_수용한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """7B 허용 오차: 구형처럼 intent(단수)만 채워 와도 계획으로 반영한다."""
    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[
                {
                    "name": "ClassifyAndRewrite",
                    "args": {"rewritten_query": "인사", "intent": "SMALLTALK"},
                }
            ],
            content="",
        )
    )
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "안녕!"})
    assert result["intents"] == ["SMALLTALK"]


def test_route_자리표시_재작성은_원본_질문으로_대체한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """재시도 예시의 자리표시(<...>)를 그대로 복사한 응답 방어."""
    fake = _FakeLLM(_classify_response("<검색용으로 재작성한 질문>", ["DOC_SEARCH"]))
    monkeypatch.setattr(router_module, "get_llm", lambda: fake)
    result = router_module.route({"question": "육아휴직 얼마나 써?"})
    assert result["rewritten_query"] == "육아휴직 얼마나 써?"
    assert result["intents"] == ["DOC_SEARCH"]


# ---------- graph: 실행 큐 분기와 도구 스텝 ----------


def test_next_step은_큐의_선두를_따른다() -> None:
    assert next_step({"pending_intents": ["DISCHARGE_DAYS", "DOC_SEARCH"]}) == "DISCHARGE_DAYS"
    assert next_step({"pending_intents": ["DOC_SEARCH"]}) == "dense_retrieve"
    assert next_step({"pending_intents": []}) == "finalize"
    # 구형 상태(큐 없음)는 계획에서 재구성
    assert next_step({"intents": ["DOC_SEARCH"]}) == "dense_retrieve"


def test_after_route_단독_잡담은_종착_노드로_간다() -> None:
    assert after_route({"intents": ["SMALLTALK"]}) == "SMALLTALK"
    assert (
        after_route(
            {
                "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
                "pending_intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
            }
        )
        == "DISCHARGE_DAYS"
    )


def test_도구_스텝은_답변을_누적하고_큐에서_자신을_지운다() -> None:
    step = graph_module._make_tool_step("DISCHARGE_DAYS", lambda state: {"final_answer": "D-100"})
    result = step(
        {
            "question": "질문",
            "pending_intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
            "tool_answers": None,
        }
    )
    assert result["tool_answers"] == [{"intent": "DISCHARGE_DAYS", "answer": "D-100"}]
    assert result["pending_intents"] == ["DOC_SEARCH"]


# ---------- finalize / fallback 합성 ----------

_TOOL_ANSWERS = [{"intent": "DISCHARGE_DAYS", "answer": "전역일까지 D-100, 100일 남았습니다."}]


def test_finalize는_계획_순서로_합성한다() -> None:
    state = {
        "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
        "tool_answers": _TOOL_ANSWERS,
        "draft_answer": "전역 신청은 인사담당 부서에 합니다.",
    }
    assert finalize(state)["final_answer"] == (
        "전역일까지 D-100, 100일 남았습니다.\n\n전역 신청은 인사담당 부서에 합니다."
    )
    # 계획 순서가 반대면 합성 순서도 반대 (질문에 등장한 순서 존중)
    state["intents"] = ["DOC_SEARCH", "DISCHARGE_DAYS"]
    assert finalize(state)["final_answer"] == (
        "전역 신청은 인사담당 부서에 합니다.\n\n전역일까지 D-100, 100일 남았습니다."
    )


def test_finalize_도구_단독_계획은_도구_답변만_확정한다() -> None:
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


# ---------- generate: 도구 처리분 중복 답변 방지 ----------


def test_generate는_도구가_처리한_요청을_답하지_말라고_안내한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """복합 계획에서 도구 처리분을 generate가 창작하면 verify가 문서 파트
    전체를 탈락시킨다 (E2E 실측). 유형 설명만 넣고 수치는 넣지 않는다."""
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
    assert (
        "D-140" not in user_text and "140일" not in user_text
    )  # 수치는 미포함 (규칙 검증 오탐 방지)


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


def test_verify는_도구_처리분을_판정_범위에서_제외하라고_안내한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify가 도구 몫(답변에 없는 부분)을 이유로 문서 파트를 탈락시키는
    오탐 방지 (E2E 실측). 수치는 넣지 않고 유형 설명만 전달한다."""
    from ax_rag.query_graph.nodes import verify as verify_module

    fake = _FakeLLM(
        SimpleNamespace(
            tool_calls=[
                {"name": "VerifyAnswer", "args": {"grounded": True, "reason": "문서에 근거함"}}
            ],
            content="",
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
    assert "검증\n대상이 아니다" in user_text or "검증 대상이 아니다" in user_text.replace(
        "\n", " "
    )
    assert "D-140" not in user_text  # 도구 수치는 미포함


# ---------- main: 계획 기반 status 안내 ----------


def test_status_계획_선두가_도구면_도구별_문구를_안내한다() -> None:
    """stage="tool" + 레지스트리(TOOL_STATUS_MESSAGES) 문구 — 도구 추가 시 자동 반영."""
    assert main._status_after_node(
        "route",
        {
            "intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
            "pending_intents": ["DISCHARGE_DAYS", "DOC_SEARCH"],
        },
    ) == ("tool", "전역일을 계산하는 중...")


def test_status_계획_선두가_검색이면_검색_안내() -> None:
    assert main._status_after_node(
        "route", {"intents": ["DOC_SEARCH"], "pending_intents": ["DOC_SEARCH"]}
    ) == ("retrieve", "군 내부 문서를 검색하는 중...")


def test_status_미등록_도구는_기본_문구로_안내한다() -> None:
    """레지스트리에 문구가 없는 도구도 status가 끊기지 않는다 (도구 교체 대비)."""
    assert main._status_after_node("route", {"pending_intents": ["DISCHARGE_DAYS"]})[0] == "tool"
    # TOOL_NODES에 없는 미지 값은 검색 안내로 폴백 (fail-safe)
    assert main._status_after_node("route", {"pending_intents": ["NO_SUCH_TOOL"]}) == (
        "retrieve",
        "군 내부 문서를 검색하는 중...",
    )


def test_status_도구_완료_후_남은_큐_기준으로_안내한다() -> None:
    assert main._status_after_node("DISCHARGE_DAYS", {"pending_intents": ["DOC_SEARCH"]}) == (
        "retrieve",
        "군 내부 문서를 검색하는 중...",
    )
    assert main._status_after_node("DISCHARGE_DAYS", {"pending_intents": []}) is None

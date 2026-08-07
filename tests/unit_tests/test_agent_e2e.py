"""ReAct 루프 E2E 유닛 테스트 (컴파일된 그래프를 가짜 LLM으로 완주시킨다).

노드 단위 테스트는 "각 조각이 맞다"만 보장한다. 루프가 실제로 닫히는지,
재검색이 답변까지 이어지는지는 그래프를 돌려 봐야 알 수 있다.
외부 서비스는 전부 대체하므로 유닛 테스트로 돈다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph import graph as graph_module
from ax_rag.query_graph.nodes import act as act_module
from ax_rag.query_graph.nodes import agent as agent_module
from ax_rag.query_graph.nodes import generate as generate_module
from ax_rag.query_graph.nodes import smalltalk as smalltalk_module
from ax_rag.query_graph.nodes import verify as verify_module
from ax_rag.shared.config import get_config


class _ScriptedLLM:
    """호출 순서대로 미리 정한 응답을 돌려주는 가짜 LLM."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools: list, **kwargs: Any) -> _ScriptedLLM:
        return self

    def bind(self, **kwargs: Any) -> _ScriptedLLM:
        return self

    def invoke(self, messages: list) -> Any:
        self.calls += 1
        if not self.responses:
            raise AssertionError("예정보다 많은 LLM 호출이 발생했다")
        return self.responses.pop(0)


def _json_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[], content=json.dumps(payload, ensure_ascii=False))


def _action(action: str, thought: str = "이유", query: str = "") -> SimpleNamespace:
    return _json_response({"thought": thought, "action": action, "query": query, "content": ""})


def _text(content: str) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[], content=content)


def _verdict(grounded: bool, reason: str = "판정") -> SimpleNamespace:
    return _json_response({"grounded": grounded, "reason": reason, "unsupported": []})


def _chunk(text: str) -> dict:
    return {"text": text, "source_doc": "휴가규정.md", "rerank_score": 0.9}


def _patch_export(monkeypatch: pytest.MonkeyPatch, node: Any) -> None:
    """export_hwp의 실행 함수만 교체한다 (AgentTool은 frozen이라 통째로 갈아 끼운다)."""
    import dataclasses

    replaced = dataclasses.replace(act_module.AGENT_TOOLS["export_hwp"], node=node)
    monkeypatch.setitem(act_module.AGENT_TOOLS, "export_hwp", replaced)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """노드별 가짜 LLM과 검색을 꽂는다."""

    def _wire(
        agent_responses: list[Any],
        generate_responses: list[Any],
        verify_responses: list[Any],
        search_results: list[list[dict]],
        smalltalk_response: Any = None,
    ) -> dict[str, Any]:
        searched: list[str] = []

        def _fake_search(state: dict, query: str) -> list[dict]:
            searched.append(query)
            return search_results.pop(0) if search_results else []

        # ★ 인스턴스를 한 번만 만든다. 호출마다 새로 만들면 응답 큐가 매번
        # 처음으로 되감겨, 재시도 시나리오가 조용히 첫 응답만 반복하게 된다
        agents = _ScriptedLLM(agent_responses)
        generators = _ScriptedLLM(generate_responses)
        verifiers = _ScriptedLLM(verify_responses)
        talkers = _ScriptedLLM([smalltalk_response] if smalltalk_response is not None else [])
        monkeypatch.setattr(agent_module, "get_llm", lambda: agents)
        monkeypatch.setattr(generate_module, "get_llm", lambda: generators)
        monkeypatch.setattr(verify_module, "get_llm", lambda: verifiers)
        monkeypatch.setattr(smalltalk_module, "get_llm", lambda: talkers)
        monkeypatch.setattr(act_module, "run_search", _fake_search)
        return {"searched": searched, "agent_llm": agents}

    return _wire


@pytest.fixture(autouse=True)
def _agent_mode(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """이 파일은 ReAct 배선을 검증한다 — 배포 스위치(AGENT_MODE)와 무관하게 켠다."""
    monkeypatch.setenv("AGENT_MODE", "true")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    get_config.cache_clear()
    yield
    get_config.cache_clear()


# 모듈 전역 graph는 기동 시점의 AGENT_MODE로 컴파일된다. 스위치를 꺼 둔 환경에서도
# ReAct 루프를 검증할 수 있게 여기서는 에이전트 배선을 직접 컴파일해 쓴다
_AGENT_GRAPH = graph_module._build_agent_graph().compile()


def _invoke(question: str, **extra: Any) -> dict:
    return _AGENT_GRAPH.invoke({"question": question, "user_department": "", **extra})


def test_검색_한_번으로_답변까지_완주한다(wire) -> None:
    wire(
        agent_responses=[
            _action("search_documents", "휴가 규정을 찾는다", "연차 일수"),
            _action("finish", "근거를 모았다"),
        ],
        generate_responses=[_text("연차는 15일입니다.")],
        verify_responses=[_verdict(True)],
        search_results=[[_chunk("연차휴가는 15일이다.")]],
    )
    result = _invoke("연차 며칠이야?")
    assert result["final_answer"] == "연차는 15일입니다."
    assert result["grounded"] is True
    assert result["answer_mode"] == "grounded"
    assert result["search_calls"] == 1


def test_검증_반려되면_재검색해서_회복한다(wire) -> None:
    """★ ReAct 전환의 핵심 이득: 기존 구조는 같은 청크로 다시 쓰는 것뿐이었다."""
    wired = wire(
        agent_responses=[
            _action("search_documents", "일수를 찾는다", "연차 일수"),
            _action("finish", "근거를 모았다"),
            # 반려 사유를 보고 다른 측면으로 재검색
            _action("search_documents", "신청 절차를 찾는다", "연차 신청 절차"),
            _action("finish", "이제 답할 수 있다"),
        ],
        generate_responses=[
            _text("연차는 15일이고 신청은 부서장 승인."),
            _text("연차는 15일이며 신청서를 제출합니다."),
        ],
        verify_responses=[_verdict(False, "신청 절차가 근거에 없다"), _verdict(True)],
        search_results=[[_chunk("연차휴가는 15일이다.")], [_chunk("연차는 신청서를 제출한다.")]],
    )
    result = _invoke("연차 며칠이고 어떻게 신청해?")
    assert result["grounded"] is True
    assert result["search_calls"] == 2
    assert wired["searched"] == ["연차 일수", "연차 신청 절차"]
    assert result["verify_feedback_used"] is True


def test_재검색은_한_번뿐이고_그_뒤엔_기존_경로를_탄다(wire) -> None:
    """무한 루프 방지 — 되먹임을 쓴 뒤에는 generate 재시도/fallback으로 간다."""
    wire(
        agent_responses=[
            _action("search_documents", "찾는다", "연차"),
            _action("finish", "모았다"),
            _action("search_documents", "다시 찾는다", "연차 신청"),
            _action("finish", "끝낸다"),
        ],
        generate_responses=[_text("초안 1"), _text("초안 2"), _text("초안 3")],
        verify_responses=[
            _verdict(False, "근거 없음"),
            _verdict(False, "여전히 근거 없음"),
            _verdict(False, "근거 없음"),
        ],
        search_results=[[_chunk("본문 A")], [_chunk("본문 B")]],
    )
    result = _invoke("연차 며칠이고 어떻게 신청해?")
    assert result["answer_mode"] == "fallback"
    assert result["search_calls"] == 2  # 상한 안에서 멈춘다


def test_잡담은_검색_없이_직접_응답한다(wire) -> None:
    wire(
        agent_responses=[_action("finish", "문서가 필요 없는 인사다")],
        generate_responses=[],
        verify_responses=[],
        search_results=[],
        smalltalk_response=_text("안녕하세요! MARS입니다."),
    )
    result = _invoke("안녕!")
    assert result["final_answer"] == "안녕하세요! MARS입니다."
    assert not result.get("search_calls")
    assert result["grounded"] is False  # 출처가 붙지 않는다


def test_전역일_질문은_LLM_없이_도구만_돈다(wire) -> None:
    wired = wire(agent_responses=[], generate_responses=[], verify_responses=[], search_results=[])
    result = _invoke("전역일이 2030년 1월 1일인데 며칠 남았어?")
    assert wired["agent_llm"].calls == 0  # 결정적 매처 경로
    assert "D-" in result["final_answer"]
    # 문서 근거를 주장하지 않으므로 출처가 붙지 않는다 (grounded는 세워지지 않는다)
    assert not result.get("grounded")
    assert not result.get("retrieved_chunks")


def test_검색과_파일_저장이_한_요청에서_순서대로_처리된다(
    wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """지연 도구는 **검증 통과 후**에 실행된다 (fail-closed)."""
    exported: list[str] = []

    def _fake_export(state: dict) -> dict:
        exported.append(str(state.get("final_answer") or ""))
        return {
            "final_answer": "한글 문서로 만들었습니다.",
            "grounded": False,
            "retrieved_chunks": [],
            "generated_files": [
                {"name": "MARS_답변.hwpx", "url": "/files/x", "tool": "HWP_EXPORT"}
            ],
        }

    _patch_export(monkeypatch, _fake_export)
    wire(
        agent_responses=[
            _action("search_documents", "휴가 규정을 찾는다", "휴가 규정"),
            _action("finish", "모았다"),
        ],
        generate_responses=[_text("휴가는 15일입니다.")],
        verify_responses=[_verdict(True)],
        search_results=[[_chunk("휴가는 15일이다.")]],
    )
    result = _invoke("휴가 규정 찾아서 한글 파일로 저장해줘")
    # 검증 통과한 답변이 파일 입력으로 넘어갔다
    assert exported == ["휴가는 15일입니다."]
    assert result["final_answer"] == "휴가는 15일입니다.\n\n한글 문서로 만들었습니다."
    assert result["generated_files"][0]["name"] == "MARS_답변.hwpx"


def test_검증_실패시_파일을_만들지_않는다(wire, monkeypatch: pytest.MonkeyPatch) -> None:
    """★ fail-closed: 검증 못 받은 답변은 파일로 나가지 않는다."""
    exported: list[str] = []

    def _fake_export(state: dict) -> dict:
        exported.append("실행됨")
        return {"final_answer": "파일"}

    _patch_export(monkeypatch, _fake_export)
    # 반려 → 재검색 되돌림 → 재생성 → 소진까지 완주하는 데 필요한 응답 수
    wire(
        agent_responses=[
            _action("search_documents", "찾는다", "휴가 규정"),
            _action("finish", "모았다"),
            _action("finish", "더 찾을 것이 없다"),
        ],
        generate_responses=[_text("초안 1"), _text("초안 2"), _text("초안 3")],
        verify_responses=[
            _verdict(False, "근거 없음"),
            _verdict(False, "근거 없음"),
            _verdict(False, "근거 없음"),
        ],
        search_results=[[_chunk("무관한 본문")]],
    )
    result = _invoke("휴가 규정 찾아서 한글 파일로 저장해줘")
    assert exported == []  # 파일 생성 시도조차 없다
    assert result["answer_mode"] == "fallback"

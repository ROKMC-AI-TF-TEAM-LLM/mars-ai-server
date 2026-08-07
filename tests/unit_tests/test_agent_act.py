"""act 노드(ReAct 실행부)와 근거 병합 유닛 테스트.

여기서 고정하는 계약 두 가지가 이 전환의 안전선이다:
1) **예산 불변식** — 검색을 몇 번 하든 generate가 보는 근거는 RERANK_TOP_N개다
2) **보안 불변식** — 검색 범위(ACL·도메인)는 상태에서만 오고 LLM 인자로 오지 않는다
"""

from __future__ import annotations

import pytest

from ax_rag.query_graph import retrieval as retrieval_module
from ax_rag.query_graph.agent_tools import ACTION_SEARCH, MAX_OBSERVATION_CHARS
from ax_rag.query_graph.nodes import act as act_module
from ax_rag.query_graph.nodes.act import act, retry_search, run_deferred
from ax_rag.query_graph.retrieval import merge_chunks


def _chunk(text: str, score: float = 0.9, source: str = "휴가규정.md") -> dict:
    return {"text": text, "source_doc": source, "rerank_score": score}


# ---------- 예산 불변식 ----------


def test_근거_병합은_상한을_넘지_않는다() -> None:
    """★ 다중 검색에도 생성 컨텍스트가 늘지 않는 이유가 이 절단이다."""
    existing = [_chunk(f"기존{i}", 0.5) for i in range(5)]
    new = [_chunk(f"신규{i}", 0.9) for i in range(5)]
    merged = merge_chunks(existing, new, limit=5)
    assert len(merged) == 5


def test_근거_병합은_점수_높은_순으로_남긴다() -> None:
    merged = merge_chunks([_chunk("낮음", 0.2)], [_chunk("높음", 0.8)], limit=1)
    assert merged[0]["text"] == "높음"


def test_근거_병합은_같은_본문을_중복으로_쌓지_않는다() -> None:
    """같은 부모가 다른 검색에서 다시 뽑혀도 컨텍스트를 두 번 먹지 않는다."""
    merged = merge_chunks([_chunk("같은 본문")], [_chunk("같은 본문")], limit=5)
    assert len(merged) == 1


def test_검색을_두_번_해도_누적_근거는_상한_이내다(monkeypatch: pytest.MonkeyPatch) -> None:
    """예산 불변식의 E2E 확인 — act를 두 번 돌려도 근거가 불어나지 않는다."""
    monkeypatch.setattr(
        act_module, "run_search", lambda state, query: [_chunk(f"{query}-{i}") for i in range(5)]
    )
    state: dict = {"question": "질문", "retrieved_chunks": []}
    state.update(act({**state, "next_action": ACTION_SEARCH, "next_action_args": {"query": "1차"}}))
    state.update(act({**state, "next_action": ACTION_SEARCH, "next_action_args": {"query": "2차"}}))
    assert len(state["retrieved_chunks"]) <= 5
    assert state["search_calls"] == 2


# ---------- 검색 실행 ----------


def test_검색은_근거와_관측과_실행경로를_남긴다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(act_module, "run_search", lambda state, query: [_chunk("연차는 15일")])
    result = act(
        {
            "question": "연차 며칠?",
            "next_action": ACTION_SEARCH,
            "next_action_args": {"query": "연차 일수"},
            "agent_thought": "연차 규정을 찾는다",
        }
    )
    assert result["search_calls"] == 1
    assert result["searched_queries"] == ["연차 일수"]
    assert result["intents"] == ["DOC_SEARCH"]  # 합성 순서의 근거
    assert result["rewritten_query"] == "연차 일수"  # 리랭크·로그가 참조
    assert "<observation>" in result["agent_scratchpad"][0]["observation"]


def test_중복_검색어는_실행하지_않고_루프를_끝낸다(monkeypatch: pytest.MonkeyPatch) -> None:
    """같은 검색을 반복하면 결과도 같다 — LLM에 되묻지 않고 종료한다."""
    calls: list[str] = []

    def _spy(state: dict, query: str) -> list[dict]:
        calls.append(query)
        return [_chunk("본문")]

    monkeypatch.setattr(act_module, "run_search", _spy)
    result = act(
        {
            "question": "질문",
            "next_action": ACTION_SEARCH,
            "next_action_args": {"query": "휴가  규정"},  # 공백만 다른 같은 검색어
            "searched_queries": ["휴가 규정"],
        }
    )
    assert calls == []  # 검색을 돌리지 않았다
    assert result["force_finish"] is True


def test_관측은_상한을_넘지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """스크래치패드가 부풀면 16K 예산이 깨진다."""
    monkeypatch.setattr(
        act_module, "run_search", lambda state, query: [_chunk("가" * 5000) for _ in range(5)]
    )
    result = act(
        {"question": "질문", "next_action": ACTION_SEARCH, "next_action_args": {"query": "검색어"}}
    )
    observation = result["agent_scratchpad"][0]["observation"]
    # delimiter 태그를 뺀 본문 기준
    assert len(observation) <= MAX_OBSERVATION_CHARS + len("<observation>\n\n</observation>") + 1


def test_근거_0건도_관측으로_남는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """'못 찾았다'를 알려줘야 에이전트가 다른 검색어를 고를 수 있다."""
    monkeypatch.setattr(act_module, "run_search", lambda state, query: [])
    result = act(
        {"question": "질문", "next_action": ACTION_SEARCH, "next_action_args": {"query": "검색어"}}
    )
    assert "찾지 못했다" in result["agent_scratchpad"][0]["observation"]


# ---------- 보안 불변식 ----------


def test_검색_범위는_상태에서만_오고_LLM_인자는_무시된다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ LLM이 부서·도메인을 인자로 넣어도 검색 범위를 바꾸지 못한다."""
    captured: dict = {}

    def _spy(state: dict, query: str) -> list[dict]:
        captured.update(state)
        return []

    monkeypatch.setattr(act_module, "run_search", _spy)
    act(
        {
            "question": "질문",
            "user_department": "HR_TEAM",
            "requested_domain": "HR",
            "next_action": ACTION_SEARCH,
            # LLM이 밀어 넣은 값 — 실행에 반영되면 안 된다
            "next_action_args": {
                "query": "검색어",
                "user_department": "COMMAND",
                "requested_domain": "DIRECTIVE",
            },
        }
    )
    assert captured["user_department"] == "HR_TEAM"
    assert captured["requested_domain"] == "HR"


def test_run_search는_호출자_상태를_더럽히지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """검색을 여러 번 해도 이전 검색의 중간 산출물이 섞이지 않아야 한다."""
    monkeypatch.setattr(retrieval_module, "dense_retrieve", lambda s: {"dense_candidates": [1]})
    monkeypatch.setattr(retrieval_module, "bm25_retrieve", lambda s: {"bm25_candidates": []})
    monkeypatch.setattr(retrieval_module, "fuse", lambda s: {"retrieved_candidates": []})
    monkeypatch.setattr(
        retrieval_module, "rerank", lambda s: {"retrieved_chunks": [_chunk("본문")]}
    )
    state = {"question": "질문", "rewritten_query": "원래 쿼리"}
    retrieval_module.run_search(state, "새 쿼리")
    assert state == {"question": "질문", "rewritten_query": "원래 쿼리"}  # 원본 불변


# ---------- 도구 실행 ----------


def test_지연_도구는_실행하지_않고_예약만_한다() -> None:
    """검증 통과 전에 파일을 만들지 않는다 (fail-closed)."""
    result = act({"question": "질문", "next_action": "export_hwp", "next_action_args": {}})
    assert result["deferred_actions"] == ["export_hwp"]
    assert "generated_files" not in result  # 아직 아무것도 만들지 않았다
    assert "예약" in result["agent_scratchpad"][0]["observation"]


def test_즉시_도구는_실행하고_답변을_누적한다() -> None:
    result = act(
        {
            "question": "전역일이 2030년 1월 1일인데 며칠 남았어?",
            "next_action": "discharge_days",
            "next_action_args": {},
        }
    )
    assert result["tool_answers"][0]["intent"] == "DISCHARGE_DAYS"
    assert result["intents"] == ["DISCHARGE_DAYS"]
    assert "D-" in result["tool_answers"][0]["answer"]


# ---------- 검증 되먹임 / 지연 실행 ----------


def test_retry_search는_반려_사유를_관측으로_싣고_한_번만_돈다() -> None:
    result = retry_search({"question": "질문", "verify_reason": "신청 절차가 문서에 없다"})
    observation = result["agent_scratchpad"][0]["observation"]
    assert "신청 절차가 문서에 없다" in observation
    assert result["verify_feedback_used"] is True
    assert result["draft_answer"] == ""  # 반려된 초안은 버린다


def test_지연_도구는_확정_답변_뒤에_이어_붙는다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        act_module,
        "AGENT_TOOLS",
        {
            "export_hwp": type(
                "T",
                (),
                {
                    "node": staticmethod(
                        lambda state: {
                            "final_answer": "파일을 만들었습니다.",
                            "generated_files": [{"name": "a.hwpx"}],
                        }
                    )
                },
            )()
        },
    )
    result = run_deferred(
        {"final_answer": "검증 통과한 답변입니다.", "deferred_actions": ["export_hwp"]}
    )
    assert result["final_answer"] == "검증 통과한 답변입니다.\n\n파일을 만들었습니다."
    assert result["generated_files"] == [{"name": "a.hwpx"}]
    assert result["deferred_actions"] == []  # 두 번 실행되지 않는다


def test_예약이_없으면_지연_노드는_아무것도_바꾸지_않는다() -> None:
    assert run_deferred({"final_answer": "답변"}) == {}

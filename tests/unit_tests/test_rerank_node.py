"""nodes/rerank.py 유닛 테스트 — 리랭커/부모 저장소를 가짜로 대체해 로직만 검증."""

from __future__ import annotations

from typing import Any

import pytest

from ax_rag.query_graph.nodes import rerank as rerank_module
from ax_rag.shared.config import get_config

_PARENTS = {
    "p1": "부모 텍스트 1: 육아휴직은 최대 1년까지 사용할 수 있다. (전후 맥락 포함)",
    "p2": "부모 텍스트 2: 연차휴가는 매년 15일이 부여된다. (전후 맥락 포함)",
}


class _FakeResponse:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"scores": self._scores}


@pytest.fixture()
def fake_services(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """리랭커 HTTP 호출과 parent_store.get_parent를 가짜로 대체한다."""
    calls: dict[str, Any] = {"scores": []}

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:
        calls["url"] = url
        calls["timeout"] = timeout
        return _FakeResponse(calls["scores"])

    monkeypatch.setattr(rerank_module.requests, "post", fake_post)
    monkeypatch.setattr(rerank_module.parent_store, "get_parent", lambda pid: _PARENTS.get(pid, ""))
    return calls


def _candidate(chunk_id: str, parent_id: str, source_doc: str = "휴가규정.md") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"자식 청크 {chunk_id}",
        "parent_id": parent_id,
        "source_doc": source_doc,
    }


def test_점수순_top_n_확정_후_부모로_치환된다(fake_services: dict) -> None:
    candidates = [
        _candidate("c1", "p1"),
        _candidate("c2", "p2"),
        _candidate("c3", "p_없음"),
    ]
    # 전부 임계값(기본 0.5) 이상 — 정렬·부모 치환만 검증한다
    fake_services["scores"] = [0.6, 0.9, 0.7]  # c2 > c3 > c1

    result = rerank_module.rerank(
        {"question": "질문", "rewritten_query": "검색 쿼리", "retrieved_candidates": candidates}
    )
    chunks = result["retrieved_chunks"]

    assert len(chunks) == 3  # RERANK_TOP_N=5보다 후보가 적으면 전부
    assert chunks[0]["text"] == _PARENTS["p2"]  # 최고점 c2 → 부모 p2로 치환
    assert chunks[1]["text"] == "자식 청크 c3"  # 부모 조회 실패 → 자식 텍스트 폴백
    assert chunks[0]["rerank_score"] == 0.9
    assert chunks[2]["text"] == _PARENTS["p1"]  # 최저점 c1도 임계값 이상이면 포함
    assert all("source_doc" in c for c in chunks)


def test_같은_부모의_자식들은_한_번만_치환된다(fake_services: dict) -> None:
    candidates = [
        _candidate("c1", "p1"),
        _candidate("c2", "p1"),  # 같은 부모
        _candidate("c3", "p2"),
    ]
    fake_services["scores"] = [0.9, 0.8, 0.7]

    result = rerank_module.rerank(
        {"question": "질문", "rewritten_query": "검색 쿼리", "retrieved_candidates": candidates}
    )
    chunks = result["retrieved_chunks"]

    texts = [c["text"] for c in chunks]
    assert texts.count(_PARENTS["p1"]) == 1  # 부모 중복 없음
    assert _PARENTS["p2"] in texts


def test_후보가_없으면_빈_결과(fake_services: dict) -> None:
    result = rerank_module.rerank(
        {"question": "질문", "rewritten_query": "검색 쿼리", "retrieved_candidates": []}
    )
    assert result == {"retrieved_chunks": []}


def test_임계값_미만_후보는_근거에서_제외된다(fake_services: dict) -> None:
    """무관 문서(0.0x 점수)가 top_n을 채워 컨텍스트·출처를 오염시키는 것 방지.

    실측 분포: 관련 청크는 0.6+, 무관 청크는 0.05 미만으로 극명하게 갈린다.
    """
    candidates = [
        _candidate("c1", "p1", "휴가규정.md"),
        _candidate("c2", "p2", "무관문서.pdf"),
        _candidate("c3", "p_없음", "무관문서2.pdf"),
    ]
    fake_services["scores"] = [0.99, 0.04, 0.01]  # 관련 1건 + 무관 2건 (실측 패턴)

    result = rerank_module.rerank(
        {"question": "질문", "rewritten_query": "검색 쿼리", "retrieved_candidates": candidates}
    )
    chunks = result["retrieved_chunks"]
    assert len(chunks) == 1  # 기본 임계값(0.5) 미만은 탈락
    assert chunks[0]["source_doc"] == "휴가규정.md"


def test_전부_임계값_미만이면_근거_없음으로_fail_closed_유도(fake_services: dict) -> None:
    fake_services["scores"] = [0.4, 0.01]  # 애매한 점수도 기본 임계값 0.5 미만
    result = rerank_module.rerank(
        {
            "question": "질문",
            "rewritten_query": "검색 쿼리",
            "retrieved_candidates": [_candidate("c1", "p1"), _candidate("c2", "p2")],
        }
    )
    # 근거 0건 → generate가 빈 초안 → verify fail-closed → fallback 답변
    assert result["retrieved_chunks"] == []


def test_리랭커_호출에_timeout이_지정된다(fake_services: dict) -> None:
    """CLAUDE.md: 외부 서비스 호출에는 반드시 timeout 지정."""
    fake_services["scores"] = [0.5]
    rerank_module.rerank(
        {
            "question": "질문",
            "rewritten_query": "검색 쿼리",
            "retrieved_candidates": [_candidate("c1", "p1")],
        }
    )
    # 값 자체는 .env가 덮어쓸 수 있으므로 설정과 일치하는지만 검증한다
    assert fake_services["timeout"] == get_config().HTTP_TIMEOUT_SECONDS

"""HWP_DRAFT 도구 유닛 테스트 — 사용자 제공 내용의 문서 초안 작성 + HWPX 저장."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ax_rag.query_graph.graph import after_route
from ax_rag.query_graph.nodes import hwp_draft as draft_module
from ax_rag.query_graph.nodes.hwp_draft import DRAFT_FAIL_ANSWER, hwp_draft
from ax_rag.query_graph.nodes.router import _normalize_plan
from ax_rag.shared.config import get_config


class _FakeLLM:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.bind_kwargs: dict = {}
        self.captured_messages: list | None = None

    def bind(self, **kwargs: Any) -> _FakeLLM:
        self.bind_kwargs.update(kwargs)
        return self

    def invoke(self, messages: list) -> Any:
        self.captured_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.response


@pytest.fixture()
def export_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    monkeypatch.setenv("EXPORT_DIR", str(tmp_path))
    get_config.cache_clear()
    yield tmp_path
    get_config.cache_clear()


def test_초안을_작성해_hwpx로_저장하고_미리보기와_링크를_답한다(
    monkeypatch: pytest.MonkeyPatch, export_dir: Path
) -> None:
    draft_text = "1. 목적: 부대 워크숍 개최\n2. 일시: [빈칸]\n3. 장소: 대강당"
    fake = _FakeLLM(SimpleNamespace(content=draft_text))
    monkeypatch.setattr(draft_module, "get_llm", lambda: fake)

    result = hwp_draft(
        {
            "question": "부대 워크숍 공문 초안 잡아서 파일로 만들어줘. 장소는 대강당이야",
            "conversation_history": [],
        }
    )
    assert result["grounded"] is False
    assert draft_text in result["final_answer"]  # 미리보기 포함
    assert "/files/" in result["final_answer"]
    files = list(export_dir.glob("MARS_초안_*.hwpx"))
    assert len(files) == 1
    with zipfile.ZipFile(files[0]) as archive:
        section = archive.read("Contents/section0.xml").decode("utf-8")
    assert "부대 워크숍 개최" in section


def test_프롬프트는_창작_금지와_빈칸_규칙을_강제하고_설정_온도를_쓴다(
    monkeypatch: pytest.MonkeyPatch, export_dir: Path
) -> None:
    fake = _FakeLLM(SimpleNamespace(content="초안"))
    monkeypatch.setattr(draft_module, "get_llm", lambda: fake)
    hwp_draft({"question": "공문 초안 만들어 파일로 생성해줘", "conversation_history": []})

    system_text = fake.captured_messages[0].content
    assert "지어내지 않는다" in system_text  # 근거 없는 창작 금지 (verify 밖 경로 원칙)
    assert "[빈칸]" in system_text
    assert fake.bind_kwargs == {"temperature": get_config().GENERATE_TEMPERATURE}


def test_LLM_실패_시_파일_없이_안내만_한다(
    monkeypatch: pytest.MonkeyPatch, export_dir: Path
) -> None:
    fake = _FakeLLM(exc=RuntimeError("LLM 연결 실패"))
    monkeypatch.setattr(draft_module, "get_llm", lambda: fake)
    result = hwp_draft({"question": "공문 초안 파일로 만들어줘", "conversation_history": []})
    assert result["final_answer"] == DRAFT_FAIL_ANSWER
    assert list(export_dir.iterdir()) == []


def test_단독_전용_라우팅과_계획_정규화() -> None:
    assert after_route({"intents": ["HWP_DRAFT"]}) == "HWP_DRAFT"  # 단독 종착
    # 업무 검색과 섞이면 정규화가 제거한다 (초안 작성은 단독 완결 작업)
    assert _normalize_plan(["HWP_DRAFT", "DOC_SEARCH"], []) == ["DOC_SEARCH"]

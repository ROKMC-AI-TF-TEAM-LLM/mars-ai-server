"""shared/audit_log.py 유닛 테스트 — 감사 로그 JSONL 기록."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ax_rag.shared.audit_log import log_query
from ax_rag.shared.config import get_config


@pytest.fixture()
def audit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "audit" / "audit_log.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(path))
    get_config.cache_clear()
    yield path
    get_config.cache_clear()


def test_모든_필드가_JSONL로_기록된다(audit_path: Path) -> None:
    log_query(
        user_department="HR_TEAM",
        question="육아휴직 기간은?",
        domain="HR",
        sources=["휴가규정.pdf"],
        grounded=True,
    )
    log_query(
        user_department="", question="근거 없는 질문", domain="GENERAL", sources=[], grounded=False
    )

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # append 방식

    first = json.loads(lines[0])
    assert first["user_department"] == "HR_TEAM"
    assert first["question"] == "육아휴직 기간은?"
    assert first["domain"] == "HR"
    assert first["sources"] == ["휴가규정.pdf"]
    assert first["grounded"] is True
    assert first["timestamp"]  # ISO 형식 존재

    second = json.loads(lines[1])
    assert second["grounded"] is False
    assert "육아휴직" not in second["question"]  # 한글 원문 유지 확인용 대비군


def test_answer_mode로_검증_실패와_지식_답변을_구분한다(audit_path: Path) -> None:
    """grounded=False 하나로는 '정형 안내를 냈다'와 '근거 없이 LLM 지식으로
    답했다'가 구분되지 않아, 검증 안 거친 답변이 얼마나 나갔는지 추적할 수 없다."""
    log_query(
        user_department="",
        question="군인 연금은?",
        domain="ALL",
        sources=[],
        grounded=False,
        answer_mode="knowledge",
    )
    log_query(user_department="", question="잡담", domain="ALL", sources=[], grounded=False)

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["answer_mode"] == "knowledge"
    assert records[1]["answer_mode"] is None  # 도구 단독 경로는 미표시

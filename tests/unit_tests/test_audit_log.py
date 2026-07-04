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

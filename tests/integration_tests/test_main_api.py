"""main.py SSE 통합 테스트 (전체 스택 기동 필요, 기본 skip).

실행 전제: 4개 서비스 전부 기동 + 샘플 문서 적재 (L40).
DoD: text 1개 이상 → sources 1회 → done 이벤트 순서, 감사 로그 기록.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ax_rag.shared.config import get_config

pytestmark = pytest.mark.integration

_BASE_URL = "http://localhost:9000"


def _parse_sse(body: str) -> list[str]:
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def test_이벤트_순서_text_sources_done() -> None:
    response = requests.post(
        f"{_BASE_URL}/query",
        json={"question": "육아휴직은 얼마나 쓸 수 있어?", "user_department": "HR_TEAM"},
        timeout=180,
        stream=True,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("x-accel-buffering") == "no"

    events = _parse_sse(response.text)
    payloads = [json.loads(e) for e in events]
    types = [p["type"] for p in payloads]
    assert types[-1] == "done"  # 종료 신호는 {"type": "done"} (미들웨어 계약)
    assert types.count("text") >= 1
    assert types.count("sources") == 1
    assert types[-2] == "sources"  # sources는 done 직전 1회


def test_부서_누락_시_DEPT_ONLY가_노출되지_않는다() -> None:
    """user_department 누락 → visibility ALL만 검색 (제한적 폴백)."""
    response = requests.post(
        f"{_BASE_URL}/query",
        json={"question": "법인카드 한도 알려줘"},  # 경비규정.txt는 DEPT_ONLY
        timeout=180,
    )
    payloads = [json.loads(e) for e in _parse_sse(response.text)]
    sources = next(p["items"] for p in payloads if p["type"] == "sources")
    assert all(s["name"] != "경비규정.txt" for s in sources)


def test_감사_로그가_기록된다() -> None:
    audit_path = Path(get_config().AUDIT_LOG_PATH)
    before = audit_path.read_text(encoding="utf-8").count("\n") if audit_path.exists() else 0

    requests.post(
        f"{_BASE_URL}/query",
        json={"question": "연차 이월 기준 알려줘", "user_department": "HR_TEAM"},
        timeout=180,
    )

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == before + 1
    record = json.loads(lines[-1])
    assert set(record) >= {
        "timestamp",
        "user_department",
        "question",
        "domain",
        "sources",
        "grounded",
    }

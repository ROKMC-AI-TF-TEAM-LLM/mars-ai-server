"""질의 감사 로그 (JSONL append, CLAUDE.md 보안 규칙).

모든 질의에 대해 timestamp, user_department, question, domain,
sources, grounded 여부를 기록한다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def log_query(
    user_department: str,
    question: str,
    domain: str,
    sources: list[str],
    grounded: bool,
) -> None:
    """JSONL append. 경로는 config.AUDIT_LOG_PATH."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user_department": user_department,
        "question": question,
        "domain": domain,
        "sources": sources,
        "grounded": grounded,
    }
    path = Path(get_config().AUDIT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

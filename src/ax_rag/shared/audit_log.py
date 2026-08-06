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
    answer_mode: str | None = None,
) -> None:
    """JSONL append. 경로는 config.AUDIT_LOG_PATH.

    answer_mode는 답변이 만들어진 경로다 (state.answer_mode). grounded 불리언
    하나로는 "검증 실패로 정형 안내를 냈다"와 "근거 없이 LLM 지식으로 답했다"가
    구분되지 않아, 검증을 거치지 않은 답변이 얼마나 나갔는지 사후에 추적할 수
    없다. 도구 단독 경로(잡담 등)에서는 None이다.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user_department": user_department,
        "question": question,
        "domain": domain,
        "sources": sources,
        "grounded": grounded,
        "answer_mode": answer_mode,
    }
    path = Path(get_config().AUDIT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

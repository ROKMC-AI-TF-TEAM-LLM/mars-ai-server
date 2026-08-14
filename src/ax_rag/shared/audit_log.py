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
    project_id: str = "",
    tool_calls: list[dict] | None = None,
    agent_steps: int = 0,
) -> None:
    """JSONL append. 경로는 config.AUDIT_LOG_PATH.

    answer_mode는 답변이 만들어진 경로다 (state.answer_mode). grounded 불리언
    하나로는 "검증 실패로 정형 안내를 냈다"와 "근거 없이 LLM 지식으로 답했다"가
    구분되지 않아, 검증을 거치지 않은 답변이 얼마나 나갔는지 사후에 추적할 수
    없다. 도구 단독 경로(잡담 등)에서는 None이다.

    tool_calls/agent_steps는 ReAct 루프의 행동 기록이다: 어떤 도구를 어떤
    인자로 불렀는지, 몇 라운드를 돌았는지. 재검색이 실제로 회복을 만들어내는지
    (2회 검색 후 grounded=True)를 사후에 확인할 수 있는 유일한 자료이며,
    사용자에게 표시된 thought도 여기 남는다. plan-then-execute 경로에서는
    비어 있다 (키는 항상 존재 — 추가 전용 변경).

    project_id는 실제 적용된 프로젝트 범위다 (""=전사). 정규화 후 값이므로
    형식 위반으로 전사로 떨어진 요청도 ""로 기록된다.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user_department": user_department,
        "question": question,
        "domain": domain,
        "project_id": project_id,
        "sources": sources,
        "grounded": grounded,
        "answer_mode": answer_mode,
        "tool_calls": tool_calls or [],
        "agent_steps": agent_steps,
    }
    path = Path(get_config().AUDIT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

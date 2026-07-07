"""smalltalk 노드: 잡담/인사에 검색·검증 없이 직접 응답한다.

라우터가 SMALLTALK으로 분류한 입력만 온다. 문서 근거를 주장하지 않으므로
grounded=False로 두어 main.py가 sources를 붙이지 않게 한다.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    SMALLTALK_DEFAULT_ANSWER,
    SMALLTALK_SYSTEM_PROMPT,
    history_to_messages,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def smalltalk(state: QueryState) -> dict:
    """가벼운 대화 응답. 실패해도 기본 인사로 폴백해 파이프라인을 죽이지 않는다."""
    config = get_config()
    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    try:
        response = get_llm().invoke(
            [
                SystemMessage(SMALLTALK_SYSTEM_PROMPT),
                *history_to_messages(history),
                HumanMessage(state["question"]),
            ]
        )
        answer = str(response.content).strip() or SMALLTALK_DEFAULT_ANSWER
    except Exception:
        logger.exception("smalltalk 호출 실패 → 기본 인사 폴백")
        answer = SMALLTALK_DEFAULT_ANSWER

    logger.info("smalltalk 응답: %s", answer[:80])
    # 문서 근거가 없으므로 grounded=False (sources 미노출), 검색 결과도 비운다
    return {"final_answer": answer, "grounded": False, "retrieved_chunks": []}

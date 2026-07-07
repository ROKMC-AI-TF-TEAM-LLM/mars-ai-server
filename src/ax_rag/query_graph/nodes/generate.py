"""generate 노드: 검색 근거 기반 답변 생성.

프롬프트에 원본 질문과 rewritten_query를 둘 다 포함시켜
검색-생성 미스매치를 모델이 감지할 여지를 남긴다 (architecture.md §4).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    format_documents,
    history_to_messages,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def generate(state: QueryState) -> dict:
    """<document> delimiter로 감싼 근거 + 원본/재작성 질문으로 답변 초안을 만든다."""
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        # 근거가 전혀 없으면 생성하지 않는다 → verify가 fail-closed로 fallback 유도
        logger.warning("검색 근거 없음 → 빈 초안 반환")
        return {"draft_answer": ""}

    config = get_config()
    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    user_prompt = GENERATE_USER_TEMPLATE.format(
        documents=format_documents(chunks),
        question=state["question"],
        rewritten_query=state.get("rewritten_query") or state["question"],
    )
    response = get_llm().invoke(
        [
            SystemMessage(GENERATE_SYSTEM_PROMPT),
            *history_to_messages(history),
            HumanMessage(user_prompt),
        ]
    )
    draft = str(response.content).strip()
    logger.info("답변 초안 생성: %d자", len(draft))
    return {"draft_answer": draft}

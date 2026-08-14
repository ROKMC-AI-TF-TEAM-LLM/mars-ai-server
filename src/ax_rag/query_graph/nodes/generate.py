"""generate 노드: 검색 근거 기반 답변 생성.

프롬프트에 원본 질문과 rewritten_query를 둘 다 포함시켜
검색-생성 미스매치를 모델이 감지할 여지를 남긴다 (architecture.md §4).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    GENERATE_RETRY_TEMPLATE,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_TOOL_HANDLED_TEMPLATE,
    GENERATE_USER_TEMPLATE,
    format_documents,
    format_instructions,
    history_to_messages,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import format_handled_note
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _retry_note(state: QueryState) -> str:
    """재생성이면 직전 반려 사유를 재작성 지시로 만든다 (1차 생성이면 빈 문자열).

    increment_retry가 실어 보낸 retry_hint가 있을 때만 붙는다.
    """
    hint = str(state.get("retry_hint") or "").strip()
    if not hint:
        return ""
    return GENERATE_RETRY_TEMPLATE.format(reason=hint)


def generate(state: QueryState) -> dict:
    """<document> delimiter로 감싼 근거 + 원본/재작성 질문으로 답변 초안을 만든다."""
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        # 근거가 전혀 없으면 생성하지 않는다 → verify가 fail-closed로 fallback 유도
        logger.warning("검색 근거 없음 → 빈 초안 반환")
        return {"draft_answer": ""}

    config = get_config()
    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    user_prompt = (
        GENERATE_USER_TEMPLATE.format(
            documents=format_documents(chunks),
            # 지침은 근거와 질문 **사이**에 들어간다 (템플릿이 순서를 고정한다)
            instructions=format_instructions(state),
            question=state["question"],
            rewritten_query=state.get("rewritten_query") or state["question"],
        )
        + format_handled_note(state, GENERATE_TOOL_HANDLED_TEMPLATE)
        + _retry_note(state)
    )
    # 답변 생성만 설정 온도로 호출한다 (기본 0.2 — 문장 자연스러움 + 재시도
    # 다양성). 라우터·verify는 get_llm() 기본값 0 유지 (분류·판정 재현성)
    response = (
        get_llm()
        .bind(temperature=config.GENERATE_TEMPERATURE)
        .invoke(
            [
                SystemMessage(GENERATE_SYSTEM_PROMPT),
                *history_to_messages(history),
                HumanMessage(user_prompt),
            ]
        )
    )
    draft = str(response.content).strip()
    logger.info(
        "답변 초안 생성: %d자%s",
        len(draft),
        " (반려 사유 반영 재생성)" if _retry_note(state) else "",
    )
    # 검증 반려를 사후에 진단하려면 **초안 본문**이 있어야 한다. 길이만으로는
    # "지어냈는가"와 "모른다고 답했는가"를 구분할 수 없어 추측만 하게 된다 (실측 사례)
    logger.debug("초안 전문 ↓\n%s", draft)
    return {"draft_answer": draft}

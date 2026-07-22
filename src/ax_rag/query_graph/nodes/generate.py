"""generate 노드: 검색 근거 기반 답변 생성.

프롬프트에 원본 질문과 rewritten_query를 둘 다 포함시켜
검색-생성 미스매치를 모델이 감지할 여지를 남긴다 (architecture.md §4).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GENERATE_TOOL_HANDLED_TEMPLATE,
    GENERATE_USER_TEMPLATE,
    format_documents,
    history_to_messages,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tools import POST_SEARCH_TOOLS, TOOL_HANDLED_LABELS
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _tool_handled_note(state: QueryState) -> str:
    """복합 계획에서 도구가 담당하는 요청 유형을 안내하는 꼬리 프롬프트.

    이미 실행된 전처리 도구(tool_answers)뿐 아니라 **아직 실행 전인 후처리
    도구**(계획 속 POST_SEARCH_TOOLS — 파일 저장 등)도 포함한다. 안 그러면
    generate가 "그 기능은 제공하지 않는다" 같은 잘못된 사족을 붙인다 (실측).
    도구 답변의 수치는 넣지 않는다 — 초안에 섞이면 rule_based_verify가
    "근거에 없는 수치"로 오탐한다. 유형 설명(TOOL_DESCRIPTIONS)만 전달한다.
    """
    handled = [item.get("intent") for item in (state.get("tool_answers") or [])]
    handled += [
        name
        for name in (state.get("intents") or [])
        if name in POST_SEARCH_TOOLS and name not in handled
    ]
    if not handled:
        return ""
    # 라우터용 상세 설명(TOOL_DESCRIPTIONS)이 아니라 예시 없는 짧은 라벨을 쓴다
    # — 분류 예시 문구가 안내문에 실리면 7B가 답변 내용으로 착각한다 (실측)
    lines = "\n".join(f"- {TOOL_HANDLED_LABELS.get(name, name)}" for name in handled if name)
    return GENERATE_TOOL_HANDLED_TEMPLATE.format(handled=lines)


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
    ) + _tool_handled_note(state)
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
    logger.info("답변 초안 생성: %d자", len(draft))
    return {"draft_answer": draft}

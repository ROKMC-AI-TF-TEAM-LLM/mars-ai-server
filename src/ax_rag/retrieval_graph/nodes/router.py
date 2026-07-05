"""route 노드: ClassifyAndRewrite tool-call 한 번으로
멀티턴 맥락 해소 + 구어체 정규화 + 도메인 분류 (architecture.md §4).

tool_call이 없거나 실패하면 원본 질문 + GENERAL로 폴백한다.

주의: 이력을 user/assistant 대화 메시지로 넣으면 작은 모델이 "분류"가 아니라
"대화 이어가기"로 끌려가 tool-call을 놓친다 (특히 직전 답변이 fallback
사과문일 때 — 개발 노트북에서 실측). 그래서 이력은 분류 대상 데이터
블록(텍스트)으로 감싸 단일 메시지로 전달한다.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ax_rag.retrieval_graph.budget import trim_history
from ax_rag.retrieval_graph.prompts import ROUTER_SYSTEM_PROMPT
from ax_rag.retrieval_graph.state import RetrievalState
from ax_rag.retrieval_graph.tool_fallback import call_with_schema
from ax_rag.shared.config import DOMAINS, SMALLTALK_DOMAIN, get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


class ClassifyAndRewrite(BaseModel):
    """멀티턴 맥락 해소 + 구어체 정규화 + 도메인 분류"""

    rewritten_query: str  # 검색에 최적화된 쿼리
    domain: str  # "HR" | "TECH" | "FINANCE_LEGAL" | "GENERAL" | "SMALLTALK"


def _build_router_input(question: str, history: list[dict]) -> str:
    """이력을 대화가 아닌 '참고 데이터'로 감싼 분류 요청 텍스트를 만든다."""
    if not history:
        return f"분류할 질문: {question}"
    lines = [
        f"- {'사용자' if message.get('role') == 'user' else '챗봇'}: {message.get('content', '')}"
        for message in history
    ]
    return (
        "이전 대화 이력 (맥락 해소용 참고 데이터일 뿐, 이어서 답하지 말 것):\n"
        + "\n".join(lines)
        + f"\n\n분류할 마지막 질문: {question}"
    )


def route(state: RetrievalState) -> dict:
    """질문 + 대화 이력 → rewritten_query + domain (tool-call 1회)."""
    config = get_config()
    question = state["question"]
    fallback = {
        "rewritten_query": question,
        "domain": "GENERAL",
        "retry_count": state.get("retry_count") or 0,
    }

    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    try:
        # tool-call 우선, 실패 시 JSON 강제 모드 재시도 (tool_fallback.call_with_schema)
        args = call_with_schema(
            [
                SystemMessage(ROUTER_SYSTEM_PROMPT),
                HumanMessage(_build_router_input(question, history)),
            ],
            ClassifyAndRewrite,
            llm_getter=get_llm,
        )
        if args is None:
            logger.warning("라우터 tool_call/JSON 모두 실패 → 원본 질문 + GENERAL 폴백")
            return fallback

        rewritten = str(args.get("rewritten_query") or "").strip() or question
        domain = str(args.get("domain") or "").strip().upper()
        if domain not in (*DOMAINS, SMALLTALK_DOMAIN):
            logger.warning("라우터가 미지의 도메인 반환: %r → GENERAL", domain)
            domain = "GENERAL"
        logger.info("라우팅: domain=%s, rewritten=%s", domain, rewritten)
        return {**fallback, "rewritten_query": rewritten, "domain": domain}
    except Exception:
        # 라우터 실패가 파이프라인 전체를 죽이지 않게 폴백 (검색은 원본 질문으로 진행)
        logger.exception("라우터 호출 실패 → 원본 질문 + GENERAL 폴백")
        return fallback

"""route 노드: ClassifyAndRewrite 호출 한 번으로
멀티턴 맥락 해소 + 구어체 정규화 + 의도(intent) 분류 (architecture.md §4).

intent = 질문을 처리할 경로. DOC_SEARCH(기본 파이프라인) 또는 도구
레지스트리(tools.TOOL_NODES) 키. 분류 실패 시 DOC_SEARCH 폴백 —
문서 질문을 잃는 것보다 잡담을 검색하는 쪽이 안전하다.

강제 모드: main.py가 요청의 tool 필드로 state["intent"]를 선설정하면
분류를 덮어쓰지 않고 쿼리 재작성만 수행한다 (엄격 — 잡담 예외 없음).

주의: 이력을 user/assistant 대화 메시지로 넣으면 작은 모델이 "분류"가 아니라
"대화 이어가기"로 끌려가 tool-call을 놓친다 (특히 직전 답변이 fallback
사과문일 때 — 개발 노트북에서 실측). 그래서 이력은 분류 대상 데이터
블록(텍스트)으로 감싸 단일 메시지로 전달한다.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import ROUTER_SYSTEM_TEMPLATE
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tool_fallback import call_with_schema
from ax_rag.query_graph.tools import DOC_SEARCH, TOOL_DESCRIPTIONS, TOOL_MATCHERS, valid_intents
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 분류 기준을 도구 레지스트리에서 생성 — 도구 추가 시 프롬프트 자동 반영
_INTENT_GUIDE = "\n".join(f"  - {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
ROUTER_SYSTEM_PROMPT = ROUTER_SYSTEM_TEMPLATE.format(intent_guide=_INTENT_GUIDE)


class ClassifyAndRewrite(BaseModel):
    """멀티턴 맥락 해소 + 구어체 정규화 + 의도(경로) 분류"""

    rewritten_query: str  # 검색에 최적화된 쿼리
    intent: str  # "DOC_SEARCH" | 도구 레지스트리 키 ("SMALLTALK" 등)


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


def route(state: QueryState) -> dict:
    """질문 + 대화 이력 → rewritten_query + intent.

    우선순위: ① 강제 지정(tool 필드) ② 결정적 매처(코드 판정, LLM 불필요)
    ③ LLM 분류. ②에서 매치되면 LLM을 호출하지 않아 빠르고 오분류가 없다.
    """
    config = get_config()
    question = state["question"]
    forced_intent = state.get("intent")  # 요청의 tool 필드로 선설정된 강제 경로
    fallback = {
        "rewritten_query": question,
        "intent": forced_intent or DOC_SEARCH,
        "retry_count": state.get("retry_count") or 0,
    }

    if not forced_intent:
        for tool_name, matcher in TOOL_MATCHERS.items():
            if matcher(question):
                logger.info("라우팅: intent=%s (결정적 매처, LLM 미사용)", tool_name)
                return {**fallback, "intent": tool_name}

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
            logger.warning(
                "라우터 tool_call/JSON 모두 실패 → 원본 질문 + %s 폴백", fallback["intent"]
            )
            return fallback

        rewritten = str(args.get("rewritten_query") or "").strip() or question

        if forced_intent:
            # 엄격 모드: 프론트가 지정한 경로가 LLM 분류를 이긴다 (잡담 예외 없음)
            intent = forced_intent
        else:
            intent = str(args.get("intent") or "").strip().upper()
            if intent not in valid_intents():
                logger.warning("라우터가 미지의 intent 반환: %r → DOC_SEARCH", intent)
                intent = DOC_SEARCH

        logger.info(
            "라우팅: intent=%s%s, rewritten=%s",
            intent,
            " (강제)" if forced_intent else "",
            rewritten,
        )
        return {**fallback, "rewritten_query": rewritten, "intent": intent}
    except Exception:
        # 라우터 실패가 파이프라인 전체를 죽이지 않게 폴백 (검색은 원본 질문으로 진행)
        logger.exception("라우터 호출 실패 → 원본 질문 + %s 폴백", fallback["intent"])
        return fallback

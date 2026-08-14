"""knowledge_answer 노드의 본문 생성: 검색 근거가 0건일 때 LLM 자체 지식으로 답한다.

**이 경로는 verify를 거치지 않는다.** 문서 근거가 없으므로 "근거에 있는가"를
검증할 대상 자체가 없기 때문이다. 대신 두 겹으로 위험을 표시한다:
1) 프롬프트가 문서 인용을 흉내 내지 못하게 막는다 (KNOWLEDGE_ANSWER_SYSTEM_PROMPT)
2) 답변에 SSE notice 이벤트로 "문서 근거 없음" 경고가 따라붙는다 (api/pipeline.py)

발동 조건 판정과 도구 답변 합성은 graph.py가 맡는다 (fallback과 같은 구조) —
그래야 노드가 그래프를 역참조하지 않는다.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    KNOWLEDGE_ANSWER_SYSTEM_PROMPT,
    format_instructions,
    history_to_messages,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 지식 답변 전용 온도. 검증이 없는 경로라 창작을 부추기지 않도록 낮게 두되,
# 정형 문구가 아닌 대화형 답변이므로 0보다는 올린다.
# (참고: GENERATE_TEMPERATURE를 0.2→0.3으로 올렸을 때 근거 밖 문장이 늘어난
#  실측이 있다 — config.GENERATE_TEMPERATURE 주석 참조)
_KNOWLEDGE_TEMPERATURE = 0.3


def generate_knowledge_answer(state: QueryState) -> str:
    """문서 근거 없이 LLM 지식으로 답변 본문을 만든다. 실패하면 빈 문자열.

    빈 문자열이면 호출부(graph.knowledge_answer)가 정형 fallback으로 되돌린다 —
    경고 이벤트가 붙지 않은 채 빈 답변이 나가지 않게 하기 위해서다.
    """
    config = get_config()
    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    try:
        response = (
            get_llm()
            .bind(temperature=_KNOWLEDGE_TEMPERATURE)
            .invoke(
                [
                    SystemMessage(KNOWLEDGE_ANSWER_SYSTEM_PROMPT),
                    *history_to_messages(history),
                    HumanMessage(format_instructions(state) + state["question"]),
                ]
            )
        )
    except Exception:
        logger.exception("지식 기반 답변 생성 실패 → 정형 안내로 폴백")
        return ""

    answer = str(response.content).strip()
    if not answer:
        logger.warning("지식 기반 답변이 비어 있다 → 정형 안내로 폴백")
        return ""
    logger.info("지식 기반 답변 생성: %d자 (문서 근거 없음, 검증 미거침)", len(answer))
    return answer

"""verify 노드: LLM 근거 검증 (architecture.md §4).

VerifyAnswer 도구 호출로 draft_answer가 retrieved_chunks에 근거하는지 판정한다.

fail-closed: 판정 불가(빈 답변, 근거 0건, tool_call 부재, 예외)면 grounded=False.
검증 실패를 통과시키는 코드를 만들지 않는다 (CLAUDE.md).

규칙 기반 1차 검증(수치·문서명이 근거에 문자열로 있는지)은 제거했다.
부분 문자열 대조라 정밀도가 낮았고("150"이 "1500"에 매칭), 종합 표현을
살리려 예외 경로를 덧대는 과정에서 오탐·오통과가 반복됐다
(목록 번호, "6개월"의 개수 오인, 2026+1=2027로 지어낸 연도 통과).
LLM 검증이 그 몫을 온전히 대신하는 것을 실측으로 확인했다 —
지어낸 일수/이월 한도/기한/조항 번호 4종을 3회씩 전부 grounded=False로
판정했고(15/15), 규칙이 못 하던 "근거는 15인데 25라고 했다"는 **모순**까지 잡는다.
"""

from __future__ import annotations

from typing import ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ax_rag.query_graph.prompts import (
    VERIFY_SYSTEM_PROMPT,
    VERIFY_TOOL_HANDLED_TEMPLATE,
    VERIFY_USER_TEMPLATE,
    format_documents,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tool_fallback import call_with_schema
from ax_rag.query_graph.tools import format_handled_note
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


class VerifyAnswer(BaseModel):
    """답변이 문서에 근거하는지 검증"""

    grounded: bool
    reason: str

    # JSON 강제 재시도용 형식 예시 (tool_fallback._retry_example).
    # grounded 예시가 False인 이유: 앵무새 복사돼도 fail-closed로 떨어진다
    RETRY_EXAMPLE: ClassVar[dict] = {"grounded": False, "reason": "<판단 근거 한 문장>"}


def check_preconditions(draft_answer: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """LLM에 판정을 물을 수 있는 상태인지 확인한다. (가능 여부, 사유) 반환.

    답변이나 근거가 아예 없으면 LLM에 물을 것이 없다 — 호출을 아끼고
    곧바로 fail-closed로 떨어뜨린다. 내용의 근거 여부는 판단하지 않는다.
    """
    if not draft_answer.strip():
        return False, "답변이 비어 있다"
    if not retrieved_chunks:
        return False, "검증할 근거 청크가 없다"
    return True, "검증 가능"


def verify(state: QueryState) -> dict:
    """LLM 근거 검증. 판정 불가 시 grounded=False (fail-closed)."""
    draft = state.get("draft_answer") or ""
    chunks = state.get("retrieved_chunks") or []

    ready, reason = check_preconditions(draft, chunks)
    if not ready:
        logger.warning("검증 전제 미충족: %s", reason)
        return {"grounded": False, "verify_reason": f"검증 전제 미충족: {reason}"}

    try:
        # tool-call 우선, 실패 시 JSON 강제 모드 재시도 (tool_fallback.call_with_schema)
        args = call_with_schema(
            [
                SystemMessage(VERIFY_SYSTEM_PROMPT),
                HumanMessage(
                    VERIFY_USER_TEMPLATE.format(
                        documents=format_documents(chunks),
                        question=state["question"],
                        draft_answer=draft,
                    )
                    + format_handled_note(state, VERIFY_TOOL_HANDLED_TEMPLATE)
                ),
            ],
            VerifyAnswer,
            llm_getter=get_llm,
        )
        if args is None:
            logger.warning("검증 tool_call/JSON 모두 실패 → grounded=False (fail-closed)")
            return {"grounded": False, "verify_reason": "검증 tool_call 부재 (fail-closed)"}

        grounded = bool(args.get("grounded", False))
        reason = str(args.get("reason", "")) or "사유 없음"
        logger.info("LLM 검증: grounded=%s, reason=%s", grounded, reason)
        return {"grounded": grounded, "verify_reason": reason}
    except Exception:
        logger.exception("검증 호출 실패 → grounded=False (fail-closed)")
        return {"grounded": False, "verify_reason": "검증 호출 실패 (fail-closed)"}

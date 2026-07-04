"""verify 노드: 이중 검증 (architecture.md §4).

1차 규칙 기반: draft_answer의 수치/날짜/문서명이 retrieved_chunks에 실재하는지.
2차 LLM 기반: VerifyAnswer tool-call.

fail-closed: 어느 단계든 판정 불가(빈 답변, tool_call 부재, 예외)면 grounded=False.
검증 실패를 통과시키는 코드를 만들지 않는다 (CLAUDE.md).
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ax_rag.retrieval_graph.prompts import (
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_TEMPLATE,
    format_documents,
)
from ax_rag.retrieval_graph.state import RetrievalState
from ax_rag.retrieval_graph.tool_fallback import call_with_schema
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


class VerifyAnswer(BaseModel):
    """답변이 문서에 근거하는지 검증"""

    grounded: bool
    reason: str


# 수치(날짜 구성 요소 포함): 1,000 / 15 / 2026 / 3.5 등
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# 문서명: 확장자를 가진 파일명 패턴
_DOCNAME_RE = re.compile(
    r"[0-9A-Za-z가-힣_\-.]+\.(?:pdf|docx|doc|hwp|hwpx|md|txt|xlsx|pptx)", re.IGNORECASE
)


def _normalize_numbers(text: str) -> str:
    """천 단위 콤마 차이를 무시하기 위한 정규화."""
    return text.replace(",", "")


def rule_based_verify(draft_answer: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """1차 규칙 검증: draft_answer에 등장하는 숫자, 날짜, 문서명이
    retrieved_chunks 텍스트에 실재하는지 확인. (통과 여부, 사유) 반환.
    실패하면 LLM 검증 없이 즉시 grounded=False."""
    if not draft_answer.strip():
        return False, "답변이 비어 있다"
    if not retrieved_chunks:
        return False, "검증할 근거 청크가 없다"

    corpus = " ".join(
        f"{chunk.get('text', '')} {chunk.get('source_doc', '')}" for chunk in retrieved_chunks
    )
    corpus_normalized = _normalize_numbers(corpus)

    # 부분 문자열 검사라 "150"이 "1500"에 매칭되는 근사치이지만,
    # 근거에 아예 없는 수치/날짜(예: 지어낸 "3년")는 확실히 걸러낸다
    for number in _NUMBER_RE.findall(draft_answer):
        if _normalize_numbers(number) not in corpus_normalized:
            return False, f"근거에 없는 수치/날짜: {number}"

    for doc_name in _DOCNAME_RE.findall(draft_answer):
        if doc_name not in corpus:
            return False, f"근거에 없는 문서명: {doc_name}"

    return True, "규칙 검증 통과"


def verify(state: RetrievalState) -> dict:
    """규칙 검증 → LLM 검증. 판정 불가 시 grounded=False (fail-closed)."""
    draft = state.get("draft_answer") or ""
    chunks = state.get("retrieved_chunks") or []

    rule_ok, rule_reason = rule_based_verify(draft, chunks)
    if not rule_ok:
        logger.warning("규칙 검증 실패: %s", rule_reason)
        return {"grounded": False, "verify_reason": f"규칙 검증 실패: {rule_reason}"}

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

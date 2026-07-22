"""verify 노드: 이중 검증 (architecture.md §4).

1차 규칙 기반: draft_answer의 수치/날짜/문서명이 retrieved_chunks에 실재하는지.
2차 LLM 기반: VerifyAnswer tool-call.

fail-closed: 어느 단계든 판정 불가(빈 답변, tool_call 부재, 예외)면 grounded=False.
검증 실패를 통과시키는 코드를 만들지 않는다 (CLAUDE.md).
"""

from __future__ import annotations

import re
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
from ax_rag.query_graph.tools import POST_SEARCH_TOOLS, TOOL_HANDLED_LABELS
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def _tool_handled_note(state: QueryState) -> str:
    """복합 계획에서 도구가 처리한 요청 유형을 verify 판정 범위에서 제외하는 안내.

    도구 답변의 수치는 넣지 않는다 (검증 기준 오염 방지) — 유형 설명만 전달한다.
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
    # — 분류 예시 문구("해병대 조사해서 문서로 만들어줘")가 안내문에 실리면
    # 7B 검증기가 답변 내용으로 착각해 grounded=false 오탐을 낸다 (실측)
    lines = "\n".join(f"- {TOOL_HANDLED_LABELS.get(name, name)}" for name in handled if name)
    return VERIFY_TOOL_HANDLED_TEMPLATE.format(handled=lines)


class VerifyAnswer(BaseModel):
    """답변이 문서에 근거하는지 검증"""

    grounded: bool
    reason: str

    # JSON 강제 재시도용 형식 예시 (tool_fallback._retry_example).
    # grounded 예시가 False인 이유: 앵무새 복사돼도 fail-closed로 떨어진다
    RETRY_EXAMPLE: ClassVar[dict] = {"grounded": False, "reason": "<판단 근거 한 문장>"}


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


def verify(state: QueryState) -> dict:
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
                    + _tool_handled_note(state)
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

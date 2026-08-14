"""verify 노드: LLM 근거 검증 (architecture.md §4).

VerifyAnswer 도구 호출로 draft_answer가 retrieved_chunks에 근거하는지 판정한다.

fail-closed: 판정 불가(빈 답변, 근거 0건, tool_call 부재, 예외)면 grounded=False.
검증 실패를 통과시키는 코드를 만들지 않는다 (CLAUDE.md).

규칙 기반 1차 검증(수치 문자열 대조)은 제거했다 — 정밀도가 낮아 오탐·오통과가
반복됐고, LLM 검증이 그 몫을 온전히 대신하는 것을 확인했다
(docs/experiments.md 실험 4).
"""

from __future__ import annotations

import re
from typing import ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from ax_rag.query_graph.prompts import (
    PARTIAL_ANSWER_NOTICE,
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
    unsupported: list[str] = []  # 근거 없는 문장을 그대로 복사한 목록 (부분 수용용)

    # grounded 예시가 False인 이유: 앵무새 복사돼도 fail-closed로 떨어진다
    RETRY_EXAMPLE: ClassVar[dict] = {
        "grounded": False,
        "reason": "<판단 근거 한 문장>",
        "unsupported": ["<답변에서 그대로 복사한 근거 없는 문장>"],
    }

    @field_validator("unsupported", mode="before")
    @classmethod
    def _coerce_unsupported(cls, value: object) -> object:
        """문자열 하나로 오면 리스트로 보정한다 (7B 허용 오차)."""
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


# 부분 수용 안전값
_MIN_FRAGMENT_CHARS = 8  # 이보다 짧은 지목은 무시 (조각 매칭으로 본문이 깎인다)
_MIN_REMAINING_RATIO = 0.3  # 제거 후 이 비율 미만만 남으면 부분 수용을 포기한다


def prune_unsupported(draft: str, unsupported: list[str]) -> str:
    """근거 없다고 지목된 구절을 초안에서 덜어낸다. 덜어낼 게 없으면 원본 그대로.

    **정확히 일치하는 문자열만** 지운다 — 어림짐작으로 지우면 멀쩡한 문장을 잃는다.
    너무 많이 잘리면 포기하고 원본을 돌려준다 (재생성·fallback이 맞는 상황이다).
    """
    pruned = draft
    for fragment in unsupported:
        text = str(fragment or "").strip()
        if len(text) >= _MIN_FRAGMENT_CHARS and text in pruned:
            pruned = pruned.replace(text, "")

    # 지운 자리에 남은 빈 줄·목록 기호 정리
    pruned = re.sub(r"(?m)^\s*[-*•]\s*$", "", pruned)
    pruned = re.sub(r"\n{3,}", "\n\n", pruned).strip()

    if not pruned or len(pruned) < len(draft) * _MIN_REMAINING_RATIO:
        return draft
    return pruned


def check_preconditions(draft_answer: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """LLM에 판정을 물을 수 있는 상태인지 확인한다. (가능 여부, 사유) 반환."""
    if not draft_answer.strip():
        return False, "답변이 비어 있다"
    if not retrieved_chunks:
        return False, "검증할 근거 청크가 없다"
    return True, "검증 가능"


def _judge(state: QueryState, draft: str, chunks: list[dict]) -> dict | None:
    """검증 LLM 1회 호출. 실패 시 None (호출부가 fail-closed로 처리)."""
    return call_with_schema(
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


def verify(state: QueryState) -> dict:
    """LLM 근거 검증. 판정 불가 시 grounded=False (fail-closed).

    반려 시 **부분 수용**을 한 번 시도한다: 지목된 근거 없는 문장만 덜어내고
    재검증해, 통과하면 그 정제본을 확정한다. 일부 창작 때문에 답변 전체가
    버려지는 것을 줄이되, **정제본도 반드시 재검증을 통과해야 한다.**
    """
    draft = state.get("draft_answer") or ""
    chunks = state.get("retrieved_chunks") or []

    ready, reason = check_preconditions(draft, chunks)
    if not ready:
        logger.warning("검증 전제 미충족: %s", reason)
        return {"grounded": False, "verify_reason": f"검증 전제 미충족: {reason}"}

    try:
        args = _judge(state, draft, chunks)
        if args is None:
            logger.warning("검증 tool_call/JSON 모두 실패 → grounded=False (fail-closed)")
            return {"grounded": False, "verify_reason": "검증 tool_call 부재 (fail-closed)"}

        reason = str(args.get("reason", "")) or "사유 없음"
        if bool(args.get("grounded", False)):
            logger.info("LLM 검증: grounded=True, reason=%s", reason)
            return {"grounded": True, "verify_reason": reason}

        logger.info("LLM 검증: grounded=False, reason=%s", reason)

        # ── 부분 수용: 지목된 문장만 덜어내고 재검증 ──────────────────────
        unsupported = [str(item) for item in (args.get("unsupported") or [])]
        if not unsupported:
            # ★ 진단 신호: "근거 없다"면서 정작 문장을 하나도 못 지목했다면,
            # 답변 안이 아니라 **밖**(질문의 못 답한 부분)을 보고 반려했을 가능성이 높다
            logger.info(
                "검증기가 근거 없는 문장을 지목하지 못했다 (unsupported 비어 있음) "
                "→ 부분 수용 불가. 커버리지 오판일 수 있으니 초안 전문과 함께 확인할 것"
            )
            return {"grounded": False, "verify_reason": reason}
        logger.debug("검증기가 지목한 근거 없는 문장 %d건: %s", len(unsupported), unsupported)

        pruned = prune_unsupported(draft, unsupported)
        logger.debug("부분 수용 정제본 (%d자 → %d자) ↓\n%s", len(draft), len(pruned), pruned)
        if pruned == draft:
            logger.info("부분 수용 불가: 지목 %d건이 초안과 일치하지 않는다", len(unsupported))
            return {"grounded": False, "verify_reason": reason}

        recheck = _judge(state, pruned, chunks)
        if recheck is None or not bool(recheck.get("grounded", False)):
            logger.info("부분 수용 재검증 실패 → 종전대로 반려")
            return {"grounded": False, "verify_reason": reason}

        logger.info(
            "부분 수용: 근거 없는 %d건 제거 (%d자 → %d자) 후 재검증 통과",
            len(unsupported),
            len(draft),
            len(pruned),
        )
        return {
            "grounded": True,
            "draft_answer": pruned + PARTIAL_ANSWER_NOTICE,
            "verify_reason": f"근거 없는 {len(unsupported)}건 제거 후 통과 (원 사유: {reason})",
        }
    except Exception:
        logger.exception("검증 호출 실패 → grounded=False (fail-closed)")
        return {"grounded": False, "verify_reason": "검증 호출 실패 (fail-closed)"}

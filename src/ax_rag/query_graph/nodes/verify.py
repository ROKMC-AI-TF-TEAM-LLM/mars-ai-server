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
    # 근거 없는 문장을 답변에서 그대로 복사한 목록 (부분 수용용).
    # 채워 주지 않아도 종전대로 동작한다 — 7B가 빠뜨려도 손해가 없다
    unsupported: list[str] = []

    # JSON 강제 재시도용 형식 예시 (tool_fallback._retry_example).
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

    **정확히 일치하는 문자열만** 지운다. 검증기가 요약하거나 바꿔 쓴 지목은
    건너뛴다 — 어림짐작으로 지우면 멀쩡한 문장을 잃는다.
    너무 많이 잘려 답변이 껍데기만 남으면 부분 수용을 포기하고 원본을 돌려준다
    (그 경우는 답변 전체가 근거 밖이라는 뜻이므로 재생성·fallback이 맞다).
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
    """LLM에 판정을 물을 수 있는 상태인지 확인한다. (가능 여부, 사유) 반환.

    답변이나 근거가 아예 없으면 LLM에 물을 것이 없다 — 호출을 아끼고
    곧바로 fail-closed로 떨어뜨린다. 내용의 근거 여부는 판단하지 않는다.
    """
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

    반려된 경우 **부분 수용**을 한 번 시도한다: 검증기가 지목한 근거 없는
    문장만 덜어내고 남은 답변을 재검증해, 통과하면 그 정제본을 확정한다.
    답변의 대부분이 근거 기반인데 일부 창작 때문에 통째로 버려지던 문제를
    줄이면서(실측: 병가 한도·진단서·급여는 정확한데 "지휘관 승인"을 지어내
    전체가 fallback), **최종적으로 검증을 통과한 텍스트만 내보낸다는 원칙은
    그대로 지킨다** — 정제본도 반드시 재검증을 통과해야 한다.
    """
    draft = state.get("draft_answer") or ""
    chunks = state.get("retrieved_chunks") or []

    ready, reason = check_preconditions(draft, chunks)
    if not ready:
        logger.warning("검증 전제 미충족: %s", reason)
        return {"grounded": False, "verify_reason": f"검증 전제 미충족: {reason}"}

    try:
        # tool-call 우선, 실패 시 JSON 강제 모드 재시도 (tool_fallback.call_with_schema)
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
            # ★ 이 상태가 진단의 핵심 신호다: 검증기가 "근거 없다"면서 정작
            # 근거 없는 문장을 하나도 지목하지 못했다는 뜻이다. 답변 **안의**
            # 내용이 틀렸다면 복사해 넣었을 텐데 못 넣었다면, 답변 **밖의**
            # 무언가(질문의 못 답한 부분)를 보고 반려했을 가능성이 높다.
            # 로그가 없으면 이 구분을 사후에 할 수 없어 매번 추측하게 된다
            logger.info(
                "검증기가 근거 없는 문장을 지목하지 못했다 (unsupported 비어 있음) "
                "→ 부분 수용 불가. 커버리지 오판일 수 있으니 초안 전문과 함께 확인할 것"
            )
            return {"grounded": False, "verify_reason": reason}
        logger.debug("검증기가 지목한 근거 없는 문장 %d건: %s", len(unsupported), unsupported)

        pruned = prune_unsupported(draft, unsupported)
        logger.debug("부분 수용 정제본 (%d자 → %d자) ↓\n%s", len(draft), len(pruned), pruned)
        if pruned == draft:
            # 지목이 초안과 안 맞거나(요약·변형) 너무 많이 잘린다 → 부분 수용 포기
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

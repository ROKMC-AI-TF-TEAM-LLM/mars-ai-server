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
from ax_rag.query_graph.tools import format_handled_note
from ax_rag.shared.config import get_config
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


# 수치(날짜 구성 요소 포함): 1,000 / 15 / 2026 / 3.5 등
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# 서식용 목록 번호: 줄 시작의 "1. " / "2) " / "### 3. " / "**1. " (마크다운 헤더·볼드 포함).
# LLM이 정리형 답변에 붙이는 번호는 사실 주장이 아니므로 수치 검사에서 제외한다
# — 근거에 숫자가 없는 산문이면 목록 번호만으로 탈락하는 오탐 실측. 업계 관행
# (RAGAS 등 클레임 분해 방식)도 서식 숫자를 주장으로 취급하지 않는다.
# 2자리 상한: "2026." 같은 연도(4자리)는 서식으로 오인하지 않고 계속 검사한다
_LIST_NUMBERING_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?(?:\*{1,2})?\d{1,2}[.)]\s")
# 문서명: 확장자를 가진 파일명 패턴
_DOCNAME_RE = re.compile(
    r"[0-9A-Za-z가-힣_\-.]+\.(?:pdf|docx|doc|hwp|hwpx|md|txt|xlsx|pptx)", re.IGNORECASE
)

# ── 종합 허용 경로 (근거에 문자열로 없어도 유도 가능하면 통과) ──────────────
# 배경: 모든 수치가 근거에 문자열로 있어야 통과하던 규칙이 "총 20일"·"3가지"
# 같은 종합 표현을 전부 탈락시켰다. 탈락 → 재생성 → 모델이 인용문 수준의 짧은
# 답으로 수렴하는 것을 정성 평가에서 실측(13문항 중 7문항에서 재시도 발생).
# 아래 세 경로만 예외로 열고, 나머지는 기존대로 fail-closed를 유지한다.

# 열거 개수 단위: 답변 **자신의 열거**를 세는 단위이지 규정 값의 단위가 아니다
# (규정 값은 일·개월·년·원·조·항). 지어낸 규정 수치가 이 경로로 샐 여지가 없다.
#
# "개" 뒤의 부정 전방탐색이 핵심이다 — 이게 없으면 "6개월"의 "6개"가 열거 개수로
# 잡혀 기간 수치가 근거 검사에서 빠져나간다 (실측: 초안의 "6개월씩 두 번 사용"이
# 오탐으로 탈락하고, 반대로 지어낸 기간이 통과할 수도 있었다)
_COUNT_UNIT_RE = re.compile(r"(\d+)\s*(?:가지|종류|종|번째|째|개(?!월|년|소|국))")

# 답변 속 목록 항목: "- 항목" / "* 항목" / "1. 항목" / "2) 항목"
_LIST_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d{1,2}[.)])\s+\S")

# 문장 분리 (유도 명시 경로에서 "같은 문장"의 범위를 정한다)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

# 청크 내 조합(③) 안전 포락선
_MAX_ARITHMETIC_OPERANDS = 12  # 청크 내 수치가 이보다 많으면 조합 경로 비활성
_MIN_ARITHMETIC_VALUE = 10  # 한 자리 수는 우연 일치가 흔해 조합 대상에서 제외

# 연도는 어떤 유도 경로로도 통과시키지 않는다 — 근거에 그대로 있어야 한다.
# 실측: 근거의 "2026년"과 "1년"으로 2026+1=2027이 성립해 **지어낸 개정 연도가
# 통과**했다. 연도는 규정의 효력 시점을 좌우하는 값이라 오통과 피해가 크고,
# 합산으로 얻어지는 값도 아니다 (인접 연도가 근거에 있으면 거의 항상 유도된다)
_YEAR_MIN, _YEAR_MAX = 1900, 2100


def _normalize_numbers(text: str) -> str:
    """천 단위 콤마 차이를 무시하기 위한 정규화."""
    return text.replace(",", "")


def _as_int(token: str) -> int | None:
    """수치 토큰을 정수로 변환한다. 소수는 조합 검증 대상이 아니므로 None."""
    normalized = _normalize_numbers(token)
    if "." in normalized:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _is_year(value: int) -> bool:
    """연도로 보이는 값인지 — 유도 경로 전체에서 제외 대상."""
    return _YEAR_MIN <= value <= _YEAR_MAX


def _derivation_shown(number: str, draft: str, corpus_normalized: str) -> bool:
    """① 답변이 계산 과정을 보여준 경우 — 같은 문장의 피연산자로 검증한다.

    결과와 피연산자가 **같은 문장 안에** 있어야 한다. 모델이 근거를 제시한
    것이지 추측이 아니라는 증거이므로, 청크 내 조합(③)보다 안전하다.
    예: "연차 15일과 특별휴가 5일을 합해 총 20일" → 15+5=20, 15·5가 근거에 있음

    피연산자 크기는 제한하지 않는다 (모델이 계산을 명시했으므로 5일 같은 작은
    값도 정당하다). 다만 연도는 유도 대상이 아니다.
    """
    target = _as_int(number)
    if target is None or _is_year(target):
        return False
    for sentence in _SENTENCE_SPLIT_RE.split(draft):
        if number not in sentence:
            continue
        operands = [
            value
            for token in _NUMBER_RE.findall(sentence)
            if token != number
            and (value := _as_int(token)) is not None
            and _normalize_numbers(token) in corpus_normalized
        ]
        for i, left in enumerate(operands):
            for right in operands[i + 1 :]:
                if left + right == target or abs(left - right) == target:
                    return True
    return False


def _enumeration_note(number: str, draft: str) -> str:
    """② 열거 개수("N가지")를 수치 검사에서 제외한 사유 (로그용).

    개수가 목록 항목 수와 다르면 모델의 계수 오류지만 **탈락시키지는 않는다**.
    rule_based_verify의 책임은 "근거에 있는 사실인가"이지 "답변이 자기 내부에서
    일관된가"가 아니다. 불일치로 탈락시켰더니 정상 답변이 fallback으로 떨어지는
    것을 실측했고(초안의 "6개월"이 개수로 오인된 사고), 계수 오류는 뒤따르는
    LLM 검증이 잡는다.
    """
    item_count = len(_LIST_ITEM_RE.findall(draft))
    if item_count and _as_int(number) != item_count:
        return f"열거 개수 {number}가 목록 항목 {item_count}개와 다름 (계수 오류 가능)"
    return "열거 개수 표현"


def _chunk_arithmetic(number: str, chunk_numbers: list[list[int]]) -> str | None:
    """③ 같은 청크 안의 두 수치의 합·차로 설명되면 유도식을 반환한다 (없으면 None).

    가장 느슨한 경로라 안전 포락선을 건다: 같은 청크 내로 제한, 2항만,
    청크 내 수치가 너무 많으면 비활성, 결과·피연산자 모두 한 자리 수 제외,
    연도 제외.

    **피연산자도 하한을 두는 이유**: 근거에 1·2 같은 작은 수가 있으면
    임의의 X를 (X-1)+1로 만들 수 있어 규칙이 사실상 무력해진다 (실측:
    2026 + 1 = 2027로 지어낸 연도가 통과). 종합이란 유의미한 두 값을 합치는
    것이지 ±1 미세 조정이 아니다.
    """
    target = _as_int(number)
    if target is None or target < _MIN_ARITHMETIC_VALUE or _is_year(target):
        return None
    for numbers in chunk_numbers:
        if len(numbers) > _MAX_ARITHMETIC_OPERANDS:
            continue  # 조합 폭발 — 이 청크는 건너뛴다
        operands = [value for value in numbers if value >= _MIN_ARITHMETIC_VALUE]
        for i, left in enumerate(operands):
            for right in operands[i + 1 :]:
                if left + right == target:
                    return f"{left} + {right} = {target}"
                if abs(left - right) == target:
                    return f"{max(left, right)} - {min(left, right)} = {target}"
    return None


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

    # 목록 번호(서식)는 검사 대상 텍스트에서만 걷어낸다 — 답변 본문 속 수치는
    # 전수 검사 유지 (fail-closed). 원문(draft_answer)은 변경하지 않는다
    draft_for_numbers = _LIST_NUMBERING_RE.sub("", draft_answer)

    count_tokens = set(_COUNT_UNIT_RE.findall(draft_for_numbers))
    allow_arithmetic = get_config().VERIFY_ALLOW_ARITHMETIC
    # ③용 청크별 수치 (조합 범위를 청크 안으로 가둔다 — 코퍼스 전체 조합 금지)
    chunk_numbers = [
        [value for token in _NUMBER_RE.findall(chunk.get("text", "")) if (value := _as_int(token))]
        for chunk in retrieved_chunks
    ]

    # 부분 문자열 검사라 "150"이 "1500"에 매칭되는 근사치이지만,
    # 근거에 아예 없는 수치/날짜(예: 지어낸 "3년")는 확실히 걸러낸다
    for number in _NUMBER_RE.findall(draft_for_numbers):
        if _normalize_numbers(number) in corpus_normalized:
            continue

        # ② 열거 개수: 답변 자신의 열거를 세는 표현 (규정 값이 아니다)
        if number in count_tokens:
            logger.debug("수치 %s: %s", number, _enumeration_note(number, draft_answer))
            continue

        # ① 유도 명시: 답변이 같은 문장에서 근거 수치로 계산을 보여준 경우
        if _derivation_shown(number, draft_answer, corpus_normalized):
            logger.info("수치 %s: 답변이 계산 과정을 제시해 통과", number)
            continue

        # ③ 청크 내 조합 (가장 느슨한 경로 — 설정으로 차단 가능)
        if allow_arithmetic:
            derivation = _chunk_arithmetic(number, chunk_numbers)
            if derivation is not None:
                # 감사용: 어떤 유도로 통과했는지 남겨 사후 검토가 가능하게 한다
                logger.info("수치 %s: 근거 청크 내 조합으로 통과 (%s)", number, derivation)
                continue

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

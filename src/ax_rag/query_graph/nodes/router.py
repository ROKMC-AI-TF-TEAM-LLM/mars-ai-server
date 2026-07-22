"""route 노드: ClassifyAndRewrite 호출 한 번으로
멀티턴 맥락 해소 + 구어체 정규화 + 처리 계획(intents) 분류 (architecture.md §4).

intents = 질문을 처리할 경로 목록(계획, plan-then-execute). 대부분 1개지만,
서로 다른 처리가 필요한 요청이 섞인 복합 질문이면 질문 순서대로 여러 개를
담는다. 실행(도구 순차 → 검색 파이프라인)과 합성은 graph.py가 맡는다.
분류 실패 시 [DOC_SEARCH] 폴백 — 문서 질문을 잃는 것보다 잡담을 검색하는
쪽이 안전하다.

강제 모드: main.py가 요청의 tool 필드로 state["intent"]를 선설정하면
계획을 그 경로 하나로 고정하고 쿼리 재작성만 수행한다 (엄격 — 잡담 예외 없음).

결정적 매처: 짧은 질문(_MATCHER_ONLY_MAX_CHARS 이하)이 매처에 걸리면 LLM 없이
해당 도구 단독 계획으로 직행한다 (빠르고 오분류 없음). 긴 질문은 복합일 수
있으므로 LLM 분류를 수행하되, 매처가 잡은 도구는 계획에 반드시 포함시킨다.

주의: 이력을 user/assistant 대화 메시지로 넣으면 작은 모델이 "분류"가 아니라
"대화 이어가기"로 끌려가 tool-call을 놓친다 (특히 직전 답변이 fallback
사과문일 때 — 개발 노트북에서 실측). 그래서 이력은 분류 대상 데이터
블록(텍스트)으로 감싸 단일 메시지로 전달한다.
"""

from __future__ import annotations

import re
from typing import ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import ROUTER_SYSTEM_TEMPLATE
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tool_fallback import call_with_schema
from ax_rag.query_graph.tools import (
    DOC_SEARCH,
    POST_SEARCH_TOOLS,
    TERMINAL_ONLY_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_MATCHERS,
    execution_queue,
    valid_intents,
)
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 분류 기준을 도구 레지스트리에서 생성 — 도구 추가 시 프롬프트 자동 반영
_INTENT_GUIDE = "\n".join(f"  - {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
ROUTER_SYSTEM_PROMPT = ROUTER_SYSTEM_TEMPLATE.format(intent_guide=_INTENT_GUIDE)

# 계획 최대 길이: 한 질문에서 실행할 경로 수 상한 (비용·지연 폭주 방지)
_MAX_PLAN_STEPS = 3

# 결정적 매처 단독 종결 기준: 이 길이 이하의 질문은 복합일 가능성이 낮아
# 매처 히트 시 LLM 없이 도구 단독 계획으로 직행한다 (LLM 0회 이점 유지).
# 긴 질문은 다른 요청이 섞였을 수 있어 LLM 분류를 함께 태운다
_MATCHER_ONLY_MAX_CHARS = 30

# 검색 동반 신호: 짧은 질문이라도 이 표현이 섞이면 매처 단독 종결하지 않고
# LLM 분류를 병행한다 — "해병대 주임무 찾아서 한글 파일로 저장해줘"(28자)가
# 매처 단독으로 검색 없이 저장만 실행되는 사고 실측. 매처 도구는 계획에
# 보장 포함되므로 검색 여부만 LLM이 판단하면 된다.
# "조사·정리·요약해서 문서로" 류의 검색 선행 표현을 폭넓게 포함한다
_SEARCH_HINT_RE = re.compile(
    r"찾아|검색|알아보|알려주|조사|조회|정리|요약|참고|바탕으로|근거로|기반으로"
)

# 검색 쿼리 오염 제거: 계획이 [검색 + 파일 도구]일 때 재작성 쿼리에 남은
# 파일 생성 요청 표현을 결정적으로 걷어낸다. 실측: "해병대 관련 내용을
# 조사하여 문서로 만들어줘"는 리랭크 최고점 0.022(전멸), "해병대 관련 내용"은
# 0.738 — 도구 표현이 점수를 30배 붕괴시킨다. 프롬프트 지시만으로는 7B가
# 재작성에서 요청 표현을 못 떼는 경우가 있어 코드로 보강한다
_FILE_REQUEST_RE = re.compile(
    r"(이\s*답변\s*[을를]?\s*)?(한글\s*)?(문서|파일)\s*(로|[을를])?\s*"
    r"(만들|생성|저장|내보내|출력|뽑|변환)\w*|문서화\s*해?\w*"
)
# 파일 표현 제거 후 끝에 남는 검색 동사 꼬리("...을 조사하여")도 정리한다
_TRAILING_SEARCH_VERB_RE = re.compile(r"[을를]?\s*(조사|조회|검색|정리|요약|알아보|찾아)\w*\s*$")


def _strip_file_phrases(query: str) -> str:
    """검색 쿼리에서 파일 생성 요청 표현과 꼬리 동사를 제거한다.

    제거 결과가 너무 짧으면(검색어 실종) 원본을 유지한다 — 오염된 쿼리라도
    없는 것보다는 낫다.
    """
    cleaned = _FILE_REQUEST_RE.sub(" ", query)
    if cleaned == query:
        return query  # 파일 요청 표현이 없으면 손대지 않는다 (순수 검색어 보존)
    # 파일 표현을 걷어낸 자리에 남은 검색 동사 꼬리("…을 조사하여")를 정리한다
    cleaned = _TRAILING_SEARCH_VERB_RE.sub(" ", cleaned)
    # 끝에 남은 접속 어미·조사만 정리 (명사 일부를 깎지 않게 정확 일치로)
    cleaned = re.sub(r"(하고|하여|해서|[을를])\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.~")
    return cleaned if len(cleaned) >= 2 else query


class ClassifyAndRewrite(BaseModel):
    """멀티턴 맥락 해소 + 구어체 정규화 + 처리 계획(경로 목록) 분류

    7B 허용 오차: 작은 모델이 스키마를 살짝 어겨도(문자열 intents, 단수형
    intent) 검증에서 떨어뜨리지 않고 보정해 받는다 — 검증 실패로 3단 폴백을
    다 태우면 지연 + 원본 질문 폴백(재작성 유실)이 되기 때문이다 (실측).
    """

    rewritten_query: str  # 검색에 최적화된 쿼리
    intents: list[str] = []  # 처리 경로 목록: "DOC_SEARCH" | 도구 레지스트리 키 (보통 1개)
    intent: str = ""  # (허용 오차) 단수형 응답 수용 — intents가 비어 있으면 이 값을 쓴다

    # JSON 강제 재시도용 형식 예시 (tool_fallback._retry_example).
    # 자리표시(<...>)는 앵무새 복사를 route()가 감지해 원본 질문으로 대체한다
    RETRY_EXAMPLE: ClassVar[dict] = {
        "rewritten_query": "<검색용으로 재작성한 질문>",
        "intents": ["DOC_SEARCH"],
    }

    @field_validator("intents", mode="before")
    @classmethod
    def _coerce_intents(cls, value: object) -> object:
        """intents가 문자열 하나로 오면 리스트로 보정한다."""
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


def _normalize_plan(raw_intents: list, matched: list[str]) -> list[str]:
    """LLM 분류 결과를 실행 가능한 계획으로 정규화한다.

    미지 값 제거, 중복 제거(순서 유지), 상한(_MAX_PLAN_STEPS) 절단.
    결정적 매처가 잡은 도구는 LLM이 빠뜨려도 보장 포함한다.
    단독 전용 도구(TERMINAL_ONLY_TOOLS)는 다른 경로와 섞이면 제거한다
    — verify 밖 자유 생성을 업무 답변과 한 응답으로 합성하지 않는다.
    """
    plan: list[str] = []
    for item in [*raw_intents, *matched]:
        name = str(item or "").strip().upper()
        if name in valid_intents() and name not in plan:
            plan.append(name)
    if len(plan) > 1:
        plan = [name for name in plan if name not in TERMINAL_ONLY_TOOLS]
    return plan[:_MAX_PLAN_STEPS] or [DOC_SEARCH]


def _route_result(question: str, plan: list[str], retry_count: int) -> dict:
    """route 반환 dict. intent는 계획의 대표값(첫 항목, 로그·하위 호환용).

    계획이 [검색 + 파일 도구] 복합이면 검색 쿼리의 파일 요청 표현을 정리한다.
    """
    rewritten = question
    if DOC_SEARCH in plan and any(name in POST_SEARCH_TOOLS for name in plan):
        rewritten = _strip_file_phrases(question)
    return {
        "rewritten_query": rewritten,
        "intents": plan,
        "pending_intents": execution_queue(plan),
        "intent": plan[0],
        "retry_count": retry_count,
    }


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
    """질문 + 대화 이력 → rewritten_query + 처리 계획(intents).

    우선순위: ① 강제 지정(tool 필드 — 계획을 그 경로 하나로 고정)
    ② 결정적 매처 + 짧은 질문(코드 판정, LLM 불필요) ③ LLM 분류
    (매처 히트 도구는 계획에 보장 포함).
    """
    config = get_config()
    question = state["question"]
    retry_count = state.get("retry_count") or 0
    forced_intent = state.get("intent")  # 요청의 tool 필드로 선설정된 강제 경로

    matched = (
        []
        if forced_intent
        else [name for name, matcher in TOOL_MATCHERS.items() if matcher(question)]
    )
    if (
        matched
        and len(question) <= _MATCHER_ONLY_MAX_CHARS
        and not _SEARCH_HINT_RE.search(question)
    ):
        plan = _normalize_plan([], matched)
        logger.info("라우팅: 계획=%s (결정적 매처 단독, LLM 미사용)", plan)
        return _route_result(question, plan, retry_count)

    if forced_intent:
        fallback_plan = [forced_intent]
    else:
        # LLM 실패 시에도 매처 확정 도구는 잃지 않고, 검색은 원본 질문으로 진행
        fallback_plan = _normalize_plan([*matched, DOC_SEARCH], [])
    fallback = _route_result(question, fallback_plan, retry_count)

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
            logger.warning("라우터 tool_call/JSON 모두 실패 → 원본 질문 + %s 폴백", fallback_plan)
            return fallback

        rewritten = str(args.get("rewritten_query") or "").strip() or question
        if rewritten.startswith("<") and rewritten.endswith(">"):
            # 재시도 예시의 자리표시를 그대로 복사한 응답 → 원본 질문으로 대체
            rewritten = question

        if forced_intent:
            # 엄격 모드: 프론트가 지정한 경로가 LLM 분류를 이긴다 (잡담 예외 없음)
            plan = [forced_intent]
        else:
            raw_intents = list(args.get("intents") or [])
            if not raw_intents and args.get("intent"):
                raw_intents = [args["intent"]]  # 단수형 응답 수용 (7B 허용 오차)
            plan = _normalize_plan(raw_intents, matched)

        if DOC_SEARCH in plan and any(name in POST_SEARCH_TOOLS for name in plan):
            # LLM이 재작성에서 파일 요청 표현을 못 뗀 경우 코드로 정리 (실측:
            # "…문서로 만들어줘"가 남으면 리랭크 점수 30배 붕괴 → 검색 전멸)
            rewritten = _strip_file_phrases(rewritten)

        logger.info(
            "라우팅: 계획=%s%s, rewritten=%s",
            plan,
            " (강제)" if forced_intent else "",
            rewritten,
        )
        return {**_route_result(question, plan, retry_count), "rewritten_query": rewritten}
    except Exception:
        # 라우터 실패가 파이프라인 전체를 죽이지 않게 폴백 (검색은 원본 질문으로 진행)
        logger.exception("라우터 호출 실패 → 원본 질문 + %s 폴백", fallback_plan)
        return fallback

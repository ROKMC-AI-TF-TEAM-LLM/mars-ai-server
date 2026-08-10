"""agent 노드: 다음에 할 행동 하나를 정한다 (ReAct 루프의 판단부).

한 번의 구조화 호출로 thought(판단 근거) + action(행동) + 인자를 받는다.
thought를 첫 필드로 두면 모델이 근거를 쓰고 나서 행동을 골라 선택 품질이 올라간다.

이 노드는 답변을 쓰지 않는다 — 근거를 모으는 행동만 고른다.

결정적 안전장치 (LLM이 완전히 실패해도 나빠지지 않게):
1) 요청의 tool 필드로 강제된 경로는 LLM 없이 직행
2) 결정적 매처가 잡은 도구는 LLM 없이 확정 (짧은 질문이면 종결까지)
3) 구조화 호출 실패 시 1라운드에 한해 원본 질문으로 검색 강제
4) 라운드·검색 상한을 넘으면 LLM에 묻지 않고 종료
"""

from __future__ import annotations

from typing import ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from ax_rag.query_graph.agent_tools import (
    ACTION_FINISH,
    ACTION_SEARCH,
    AGENT_TOOLS,
    INTENT_TO_ACTION,
    PHASE_DEFERRED,
    action_guide,
    resolve_action,
    scratchpad_text,
)
from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import (
    AGENT_HISTORY_TEMPLATE,
    AGENT_SCRATCHPAD_TEMPLATE,
    AGENT_SYSTEM_TEMPLATE,
    AGENT_USER_TEMPLATE,
    DEFAULT_THOUGHT,
    DEFAULT_THOUGHTS,
)
from ax_rag.query_graph.query_hygiene import (
    MATCHER_ONLY_MAX_CHARS,
    has_search_hint,
    strip_file_phrases,
)
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.thought import sanitize_thought
from ax_rag.query_graph.tool_fallback import call_with_schema
from ax_rag.query_graph.tools import DOC_SEARCH, TOOL_MATCHERS
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 행동 목록을 레지스트리에서 생성 — 도구 추가 시 프롬프트 자동 반영
AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_TEMPLATE.format(action_guide=action_guide())


class AgentAction(BaseModel):
    """다음 행동 하나 + 그 판단 근거.

    행동별 인자는 전부 기본값을 둔다 — 폴백 경로에서 모델이 키를 빠뜨려도
    검증에 걸리지 않게 하기 위해서다 (주 경로는 json_schema가 required로 강제).
    """

    thought: str = ""  # 사용자에게 표시되는 판단 근거 (한 문장)
    action: str = ""  # action_names() 중 하나
    query: str = ""  # search_documents 전용 검색어
    content: str = ""  # draft_document 전용 본문

    # JSON 강제 재시도용 형식 예시 (tool_fallback._retry_example)
    RETRY_EXAMPLE: ClassVar[dict] = {
        "thought": "<이 행동을 고른 이유>",
        "action": "search_documents",
        "query": "<검색어>",
        "content": "",
    }

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        """행동 이름의 대소문자·공백 흔들림을 보정한다 (7B 허용 오차)."""
        if isinstance(value, str):
            return value.strip().lower()
        return value


def _decision(
    step: int, action: str, args: dict, thought: str, *, force_finish: bool = False
) -> dict:
    """agent 노드의 반환 delta를 만든다 (행동 결정 + 표시용 thought).

    ★ force_finish는 **항상** 실어 보낸다. 조건부로 넣으면 이전 라운드의 True가
    남아 루프가 조기 종료된다 (LangGraph는 delta를 병합할 뿐 지우지 않는다).
    """
    cleaned = sanitize_thought(thought) or DEFAULT_THOUGHTS.get(action, DEFAULT_THOUGHT)
    return {
        "agent_steps": step,
        "agent_thought": cleaned,
        "next_action": action,
        "next_action_args": args,
        "force_finish": force_finish,
    }


def _search_decision(step: int, query: str, thought: str, *, force_finish: bool = False) -> dict:
    """검색 행동 결정. 검색어에서 파일 요청 표현을 결정적으로 걷어낸다."""
    cleaned_query = strip_file_phrases(str(query or "").strip())
    return _decision(
        step, ACTION_SEARCH, {"query": cleaned_query}, thought, force_finish=force_finish
    )


def _matched_intents(state: QueryState) -> list[str]:
    """결정적 매처가 잡은 도구 intent 목록 (LLM 없이 코드로 판정)."""
    question = state["question"]
    return [name for name, matcher in TOOL_MATCHERS.items() if matcher(question)]


def _seed_deferred(state: QueryState, matched: list[str]) -> dict:
    """매처가 잡은 지연 도구를 실행 예약에 넣는다 (LLM이 빠뜨려도 잃지 않게)."""
    deferred = list(state.get("deferred_actions") or [])
    for intent in matched:
        action = INTENT_TO_ACTION.get(intent, "")
        tool = AGENT_TOOLS.get(action)
        if tool and tool.phase == PHASE_DEFERRED and action not in deferred:
            deferred.append(action)
    return (
        {"deferred_actions": deferred} if deferred != (state.get("deferred_actions") or []) else {}
    )


def _forced_decision(state: QueryState, step: int) -> dict | None:
    """요청의 tool 필드로 강제된 경로 (엄격 모드 — LLM 분류를 건너뛴다)."""
    forced = str(state.get("intent") or "")
    if not forced:
        return None
    if forced == DOC_SEARCH:
        return _search_decision(
            step, state["question"], "요청한 대로 문서를 검색한다", force_finish=True
        )
    action = INTENT_TO_ACTION.get(forced)
    if action is None:
        return None
    return _decision(step, action, {}, "요청한 기능을 실행한다", force_finish=True)


def _caps_exhausted(state: QueryState) -> bool:
    """라운드 상한을 다 썼는지 (LLM에 묻지 않고 종료할 조건)."""
    return (state.get("agent_steps") or 0) >= get_config().MAX_AGENT_STEPS


def _search_exhausted(state: QueryState) -> bool:
    """검색 상한을 다 썼는지."""
    return (state.get("search_calls") or 0) >= get_config().MAX_SEARCH_CALLS


def _history_block(state: QueryState) -> str:
    """대화 이력을 '참고 데이터' 블록으로 만든다.

    대화 메시지로 넣으면 작은 모델이 판단 대신 대화 이어가기로 끌려간다.
    """
    history = trim_history(state.get("conversation_history") or [], get_config().HISTORY_MAX_TOKENS)
    if not history:
        return ""
    lines = "\n".join(
        f"- {'사용자' if message.get('role') == 'user' else '챗봇'}: {message.get('content', '')}"
        for message in history
    )
    return AGENT_HISTORY_TEMPLATE.format(history=lines)


def _build_messages(state: QueryState) -> list:
    """에이전트 판단용 메시지 (시스템 + 데이터 블록 하나)."""
    scratchpad = scratchpad_text(state)
    scratchpad_block = AGENT_SCRATCHPAD_TEMPLATE.format(scratchpad=scratchpad) if scratchpad else ""
    user_text = AGENT_USER_TEMPLATE.format(
        history_block=_history_block(state),
        question=state["question"],
        scratchpad_block=scratchpad_block,
    )
    return [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(user_text)]


def agent(state: QueryState) -> dict:
    """다음 행동 하나를 정한다. 실패해도 결정적 폴백으로 진행한다."""
    step = (state.get("agent_steps") or 0) + 1
    seed: dict = {}

    if step == 1:
        forced = _forced_decision(state, step)
        if forced is not None:
            logger.info("에이전트: 강제 경로 %s (LLM 미사용)", forced["next_action"])
            return forced

        matched = _matched_intents(state)
        seed = _seed_deferred(state, matched)
        inline_matched = [
            intent
            for intent in matched
            if (tool := AGENT_TOOLS.get(INTENT_TO_ACTION.get(intent, ""))) is not None
            and tool.phase != PHASE_DEFERRED
        ]
        # 짧고 검색 신호가 없는 질문은 매처만으로 종결한다 (현행 LLM 0회 경로)
        matcher_only = (
            bool(matched)
            and len(state["question"]) <= MATCHER_ONLY_MAX_CHARS
            and not has_search_hint(state["question"])
        )
        if inline_matched:
            action = INTENT_TO_ACTION[inline_matched[0]]
            logger.info("에이전트: 결정적 매처 %s (LLM 미사용, 단독종결=%s)", action, matcher_only)
            return {
                **seed,
                **_decision(
                    step, action, {}, "요청한 계산을 먼저 처리한다", force_finish=matcher_only
                ),
            }
        if matcher_only and seed.get("deferred_actions"):
            # 지연 도구만 잡힌 짧은 요청("이 답변 저장해줘") — 검색 없이 바로 종결
            logger.info("에이전트: 지연 도구 단독 요청 (LLM 미사용)")
            return {
                **seed,
                **_decision(step, ACTION_FINISH, {}, "요청한 파일을 만든다", force_finish=True),
            }

    if _caps_exhausted(state):
        logger.info("에이전트: 라운드 상한 소진 → 종료")
        return {**seed, **_decision(step, ACTION_FINISH, {}, "확인한 근거로 답변을 정리한다")}

    messages = _build_messages(state)
    logger.debug("에이전트[%d] 입력 ↓\n%s", step, messages[-1].content)
    try:
        args = call_with_schema(messages, AgentAction, llm_getter=get_llm)
    except Exception:
        logger.exception("에이전트 판단 호출 실패")
        args = None
    logger.debug("에이전트[%d] 원시 판단: %r", step, args)

    if args is None:
        if step == 1:
            logger.warning("에이전트 판단 실패(1라운드) → 원본 질문으로 검색 폴백")
            return {
                **seed,
                **_search_decision(step, state["question"], "질문 그대로 문서를 찾는다"),
            }
        logger.warning("에이전트 판단 실패(%d라운드) → 종료", step)
        return {**seed, **_decision(step, ACTION_FINISH, {}, "")}

    thought = str(args.get("thought") or "")
    action = resolve_action(str(args.get("action") or ""))
    if not thought.strip():
        # 자주 뜨면 스키마·프롬프트 문제다 (추론 스트리밍이 사실상 꺼진 상태)
        logger.warning("에이전트[%d]가 thought를 비워 보냈다 → 기본 문구로 대체", step)
    if not action:
        logger.warning("에이전트가 미지의 행동을 냈다: %r → 종료로 처리", args.get("action"))
        return {**seed, **_decision(step, ACTION_FINISH, {}, thought)}

    if action == ACTION_SEARCH:
        if _search_exhausted(state):
            logger.info("에이전트: 검색 상한 소진 → 종료")
            return {**seed, **_decision(step, ACTION_FINISH, {}, "더 찾지 않고 답변을 정리한다")}
        query = str(args.get("query") or "").strip() or state["question"]
        logger.info("에이전트[%d]: 검색 (query=%s, thought=%s)", step, query, thought)
        return {**seed, **_search_decision(step, query, thought)}

    if action == ACTION_FINISH:
        logger.info("에이전트[%d]: 종료 (thought=%s)", step, thought)
        return {**seed, **_decision(step, ACTION_FINISH, {}, thought)}

    logger.info("에이전트[%d]: 도구 %s (thought=%s)", step, action, thought)
    return {**seed, **_decision(step, action, {"content": str(args.get("content") or "")}, thought)}

"""구조화 호출 (에이전트 행동 결정·답변 검증의 공통 경로).

**주 경로는 response_format=json_schema(문법 강제)다.** 디코딩 단계에서 스키마를
강제하므로 모델이 tool-call을 낼 능력이 없어도 구조화 출력이 보장된다.

A.X-4.0-Light는 문서가 200자만 넘어도 tool-call 대신 산문을 낸다 — 모델 자체의
한계이고 설정 문제가 아니다 (docs/experiments.md 실험 8·9).

폴백은 json_schema 미지원 서빙용으로 남긴다: tool-call → JSON 강제 재시도.
전부 실패하면 None을 반환하며 호출부의 안전장치(검증: fail-closed)가 동작한다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 본문 중 가장 바깥 중괄호 블록 (```json 펜스 유무 무관)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# 정상 출력은 짧은 JSON이므로, 모델이 질문에 장문으로 답하는 폭주를 토큰
# 수준에서 차단한다. 잘려도 재시도 단계가 이어받는다
_STRUCTURED_MAX_TOKENS = 512
_RETRY_MAX_TOKENS = 256  # 재시도는 순수 JSON만 기대하므로 더 짧게

# boolean 자리표시가 False인 이유: 작은 모델이 예시를 복사해도
# 안전한 방향(fail-closed)으로 떨어지게 한다
_PLACEHOLDERS: dict[str, object] = {
    "string": "<값>",
    "boolean": False,
    "integer": 0,
    "number": 0.0,
}


def _placeholder(prop: dict) -> object:
    """json-schema 속성 하나에 대한 자리표시 값."""
    if prop.get("type") == "array":
        return [_placeholder(prop.get("items", {"type": "string"}))]
    return _PLACEHOLDERS.get(prop.get("type"), "<값>")


def json_schema_format(schema: type[BaseModel]) -> dict:
    """pydantic 스키마를 OpenAI 호환 response_format(json_schema)으로 바꾼다.

    ★ **모든 속성을 required로 올린다.** pydantic은 기본값 있는 필드를 required에서
    빼는데, 문법 강제 디코딩에서는 그것이 관용이 아니라 **면제**가 된다 — 모델이
    안 채워도 문법이 통과시킨다. 이것 때문에 부분 수용(unsupported)과 판단 근거
    스트리밍(thought)이 죽어 있었다 (docs/experiments.md 실험 11).

    파싱은 그대로 관대하다: pydantic 기본값이 남아 있어 폴백 경로에서 키가
    빠져도 검증에 걸리지 않는다.
    """
    json_schema = schema.model_json_schema()
    json_schema["additionalProperties"] = False
    json_schema["required"] = list(json_schema.get("properties", {}))
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "strict": True, "schema": json_schema},
    }


def _retry_example(schema: type[BaseModel]) -> str:
    """재시도 프롬프트용 예시 JSON 문자열.

    스키마 정의를 그대로 보여주면 작은 모델이 타입 정의를 값처럼 복사한다.
    RETRY_EXAMPLE(ClassVar)이 있으면 쓰고, 없으면 자리표시로 합성한다.
    """
    example = getattr(schema, "RETRY_EXAMPLE", None)
    if example is None:
        props = schema.model_json_schema().get("properties", {})
        example = {name: _placeholder(prop) for name, prop in props.items()}
    return json.dumps(example, ensure_ascii=False)


def extract_tool_args(response: Any, schema: type[BaseModel]) -> dict | None:
    """응답에서 tool 인자를 얻는다. 1순위 tool_calls, 2순위 본문 JSON.

    반환 dict는 schema로 타입 검증된 값이다. 실패 시 None.
    """
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        args = tool_calls[0].get("args") or {}
        try:
            return schema.model_validate(args).model_dump()
        except ValidationError:
            logger.warning("tool_call 인자가 %s 스키마에 안 맞는다: %r", schema.__name__, args)
            return None

    content = str(getattr(response, "content", "") or "")
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        if content:
            # 원문 일부를 남겨 실패 형태를 진단할 수 있게 한다 (L40 성공률 측정 재료)
            logger.warning("본문에 JSON 블록이 없다 (%s): %.200s", schema.__name__, content)
        return None
    try:
        parsed = schema.model_validate_json(match.group(0))
    except ValidationError:
        logger.warning("본문 JSON이 %s 스키마에 안 맞는다: %.200s", schema.__name__, match.group(0))
        return None
    # json_schema(주 경로)는 tool_calls가 아니라 본문에 JSON을 담아 온다.
    # 여기 도달했다고 폴백이 도는 것이 아니다 — 폴백 진입은 call_with_schema가 warning으로 남긴다
    logger.debug("본문 JSON 파싱 성공 (%s)", schema.__name__)
    return parsed.model_dump()


def call_with_schema(
    messages: list,
    schema: type[BaseModel],
    llm_getter: Callable[[], Any],
) -> dict | None:
    """스키마 기반 구조화 호출. json_schema(문법 강제) → tool-call → JSON 강제 순.

    1차: response_format=json_schema — 디코딩이 스키마를 강제하므로 모델이
        tool-call을 못 내도 구조화 출력이 나온다 (실측 3/3, 모듈 docstring 참조)
    2차: bind_tools(tool_choice=스키마명) — json_schema 미지원 서빙 대비
    3차: response_format=json_object + **예시 JSON** (스키마 정의를 값처럼
        복사하는 앵무새 실패 방지, 실측)
    전부 실패하면 None을 반환하고, 예외는 호출부의 안전장치로 전파한다.

    출력은 _STRUCTURED_MAX_TOKENS로 상한 — 모델이 분류 대신 질문에 장문으로
    답하는 폭주를 토큰 수준에서 차단한다.

    llm_getter는 호출부의 get_llm 참조를 받는다 (테스트에서 대체 가능해야 함).
    """
    try:
        schema_llm = llm_getter().bind(
            response_format=json_schema_format(schema), max_tokens=_STRUCTURED_MAX_TOKENS
        )
        args = extract_tool_args(schema_llm.invoke(messages), schema)
        if args is not None:
            return args
        logger.warning("%s json_schema 응답이 스키마에 안 맞는다 → tool-call 시도", schema.__name__)
    except Exception:
        # json_schema 미지원 서빙(구버전 등)일 수 있다 — 아래 경로로 계속 간다
        logger.warning("%s json_schema 호출 실패 → tool-call 시도", schema.__name__, exc_info=True)

    llm = (
        llm_getter()
        .bind_tools([schema], tool_choice=schema.__name__)
        .bind(max_tokens=_STRUCTURED_MAX_TOKENS)
    )
    args = extract_tool_args(llm.invoke(messages), schema)
    if args is not None:
        return args

    logger.warning("%s tool-call 파싱 실패 → JSON 강제 모드 재시도", schema.__name__)
    retry_messages = [
        *messages,
        HumanMessage(
            "위 요청의 결과를 다른 설명 없이 JSON 객체 하나로만 출력하라. "
            f"형식 예시(예시 값을 베끼지 말고 실제 값을 채울 것): {_retry_example(schema)}"
        ),
    ]
    retry_llm = llm_getter().bind(
        response_format={"type": "json_object"}, max_tokens=_RETRY_MAX_TOKENS
    )
    return extract_tool_args(retry_llm.invoke(retry_messages), schema)

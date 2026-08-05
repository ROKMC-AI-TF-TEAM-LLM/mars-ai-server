"""구조화 호출 (라우터 분류·답변 검증의 공통 경로).

**주 경로는 response_format=json_schema(문법 강제)다.** 서빙이 디코딩 단계에서
스키마를 강제하므로 모델이 tool-call을 낼 능력이 없어도 구조화 출력이 보장된다.
llama.cpp·vLLM 모두 지원한다.

tool-call을 주 경로에서 내린 이유 (실측):
A.X-4.0-Light는 **문서가 200자만 넘어도** tool-call 대신 "VerifyAnswer 도구를
사용하여 …확인하겠습니다" 같은 산문을 낸다 (100자는 성공, 200자부터 0/3 실패).
컨텍스트 여유(12,288 중 339토큰)와 무관한 모델 자체의 한계이고, llama.cpp의
chat_template_caps는 supports_tools=true로 정상 인식하며 tool_choice 네 형태가
짧은 입력에서는 모두 동작하므로 설정 문제가 아니다.

같은 근거(3,196토큰)에서의 성공률:
    tool_choice=함수         0/3   산문 출력
    response_format=json_object  0/3   "grounded=true" 평문 (JSON 아님)
    response_format=json_schema  3/3   ✅

폴백은 json_schema 미지원 서빙을 위해 남긴다: tool-call → JSON 강제 재시도.
전부 실패하면 None을 반환하며, 호출부의 안전장치(라우터: 원본 질문 폴백,
검증: fail-closed)가 그대로 동작한다.
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

# 구조화 호출 출력 상한. 라우터/검증의 정상 출력은 짧은 JSON이므로, 모델이
# tool-call을 무시하고 질문에 장문으로 답하는 폭주(46초 생성 실측)를 토큰
# 수준에서 차단한다. 잘려도 재시도 단계가 이어받는다
_STRUCTURED_MAX_TOKENS = 512
# JSON 강제 재시도는 순수 JSON만 기대하므로 더 짧게
_RETRY_MAX_TOKENS = 256

# 스키마 타입별 자리표시 값 (_retry_example의 일반 합성용).
# boolean이 False인 이유: 작은 모델이 예시를 앵무새처럼 복사해도
# 안전한 방향(fail-closed)으로 떨어지게 한다 (VerifyAnswer.grounded 등)
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

    ClassVar(RETRY_EXAMPLE 등)는 pydantic이 알아서 제외한다.
    additionalProperties=False는 스키마 밖 필드를 막아 파싱을 단순하게 한다.
    """
    json_schema = schema.model_json_schema()
    json_schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "strict": True, "schema": json_schema},
    }


def _retry_example(schema: type[BaseModel]) -> str:
    """재시도 프롬프트용 예시 JSON 문자열.

    스키마 정의(properties JSON)를 그대로 보여주면 작은 모델이 타입 정의를
    값처럼 복사해 반환한다 (실측: {"rewritten_query": {"title": ...}}).
    예시 형태로 주면 값 채우기로 따라온다. 스키마 클래스에 RETRY_EXAMPLE
    (ClassVar dict)이 있으면 그것을 쓰고, 없으면 타입별 자리표시로 합성한다.
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

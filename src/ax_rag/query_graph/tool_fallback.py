"""tool_call 폴백 파서 (roadmap 7단계의 "JSON 파싱 방식 대체" 구현).

일부 서빙 환경(llama.cpp 등)이나 프롬프트가 길 때, 모델이 tool_call 대신
본문에 ```json 블록으로 인자를 출력하는 경우가 있다. tool_calls가 비어
있으면 본문에서 스키마에 맞는 JSON을 추출해 재활용한다.

추출·검증에 실패하면 None을 반환하며, 호출부의 기존 안전장치
(라우터: 원본+GENERAL 폴백, 검증: fail-closed)가 그대로 동작한다.
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
        return None
    try:
        parsed = schema.model_validate_json(match.group(0))
    except ValidationError:
        logger.warning("본문 JSON이 %s 스키마에 안 맞는다", schema.__name__)
        return None
    logger.info("tool_call 부재 → 본문 JSON 폴백 파싱 성공 (%s)", schema.__name__)
    return parsed.model_dump()


def call_with_schema(
    messages: list,
    schema: type[BaseModel],
    llm_getter: Callable[[], Any],
) -> dict | None:
    """스키마 기반 구조화 호출: tool-call 우선, 실패 시 JSON 강제 모드 1회 재시도.

    1차: bind_tools(tool_choice=스키마명) → tool_calls 또는 본문 JSON 파싱
    2차: response_format=json_object로 재호출 (roadmap 7단계의 "JSON 파싱 대체")
    둘 다 실패하면 None을 반환하고, 예외는 호출부의 안전장치로 전파한다.

    llm_getter는 호출부의 get_llm 참조를 받는다 (테스트에서 대체 가능해야 함).
    """
    llm = llm_getter().bind_tools([schema], tool_choice=schema.__name__)
    args = extract_tool_args(llm.invoke(messages), schema)
    if args is not None:
        return args

    logger.warning("%s tool-call 파싱 실패 → JSON 강제 모드 재시도", schema.__name__)
    properties = json.dumps(schema.model_json_schema().get("properties", {}), ensure_ascii=False)
    retry_messages = [
        *messages,
        HumanMessage(
            "위 요청의 결과를 다른 설명 없이 JSON 객체 하나로만 출력하라. "
            f"필드 스키마: {properties}"
        ),
    ]
    retry_llm = llm_getter().bind(response_format={"type": "json_object"})
    return extract_tool_args(retry_llm.invoke(retry_messages), schema)

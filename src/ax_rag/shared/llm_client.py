"""LLM 클라이언트 싱글턴.

vLLM(OpenAI 호환, localhost:8000)을 가리키는 ChatOpenAI 인스턴스를
전역에서 하나만 사용한다. 노드에서 ChatOpenAI를 직접 생성하지 말고
반드시 get_llm()을 사용할 것. 라우터/생성/검증의 역할 구분은 모델이
아니라 시스템 프롬프트로 한다 (architecture.md §2).

시연 전용: EXTERNAL_LLM_ENABLED=true면 외부 OpenAI 호환 API(OpenRouter 등)를
가리킨다. 노드 코드는 어느 쪽인지 알 필요가 없다.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """LLM 서버(기본 vLLM, 시연 시 외부 API)를 가리키는 ChatOpenAI 싱글턴을 반환한다."""
    config = get_config()
    if config.EXTERNAL_LLM_ENABLED:
        # 에어갭 예외 경로 — 기동 로그에 반드시 남겨 운영 오설정을 눈에 띄게 한다
        logger.warning(
            "외부 LLM 사용 중 (시연 전용, 에어갭 예외): %s / %s",
            config.EXTERNAL_LLM_BASE_URL,
            config.EXTERNAL_LLM_MODEL_NAME,
        )
        base_url = config.EXTERNAL_LLM_BASE_URL
        api_key = config.EXTERNAL_LLM_API_KEY
        model = config.EXTERNAL_LLM_MODEL_NAME
    else:
        base_url = config.AX_BASE_URL
        api_key = config.AX_API_KEY
        model = config.AX_MODEL_NAME
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        max_retries=1,
    )

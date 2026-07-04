"""LLM 클라이언트 싱글턴.

vLLM(OpenAI 호환, localhost:8000)을 가리키는 ChatOpenAI 인스턴스를
전역에서 하나만 사용한다. 노드에서 ChatOpenAI를 직접 생성하지 말고
반드시 get_llm()을 사용할 것. 라우터/생성/검증의 역할 구분은 모델이
아니라 시스템 프롬프트로 한다 (architecture.md §2).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from ax_rag.shared.config import get_config


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """vLLM 서버를 가리키는 ChatOpenAI 싱글턴을 반환한다."""
    config = get_config()
    return ChatOpenAI(
        base_url=config.AX_BASE_URL,
        api_key=config.AX_API_KEY,
        model=config.AX_MODEL_NAME,
        temperature=0.0,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        max_retries=1,
    )

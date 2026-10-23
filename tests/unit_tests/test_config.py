"""shared/config.py 유닛 테스트: 기본값의 에어갭 준수와 불변성."""

import dataclasses

import pytest

from ax_rag.shared.config import DOMAINS, Config, get_config


@pytest.fixture()
def config() -> Config:
    """캐시를 비운 뒤 새로 로드한 설정."""
    get_config.cache_clear()
    return get_config()


def test_기본_URL은_전부_localhost다(config: Config) -> None:
    # LLM 포트는 .env가 바꿀 수 있다 (개발 노트북에서 Ollama 11434 등 서빙 교체
    # 실험) — 에어갭의 본질은 호스트가 localhost라는 것
    assert config.AX_BASE_URL.startswith("http://localhost:")
    assert config.EMBEDDING_SERVER_URL.startswith("http://localhost:8001")
    assert config.RERANKER_SERVER_URL.startswith("http://localhost:8002")


def test_localhost가_아닌_URL은_거부된다(config: Config) -> None:
    with pytest.raises(ValueError, match="에어갭"):
        dataclasses.replace(config, EMBEDDING_SERVER_URL="http://api.example.com/embed")


def test_외부_LLM은_허용_호스트_https만_통과한다(config: Config) -> None:
    replaced = dataclasses.replace(
        config,
        EXTERNAL_LLM_ENABLED=True,
        EXTERNAL_LLM_BASE_URL="https://openrouter.ai/api/v1",
        EXTERNAL_LLM_MODEL_NAME="모델",
        EXTERNAL_LLM_API_KEY="키",
    )
    assert replaced.EXTERNAL_LLM_ENABLED
    with pytest.raises(ValueError, match="https"):
        dataclasses.replace(replaced, EXTERNAL_LLM_BASE_URL="https://api.example.com/v1")
    with pytest.raises(ValueError, match="https"):
        dataclasses.replace(replaced, EXTERNAL_LLM_BASE_URL="http://openrouter.ai/api/v1")


def test_외부_LLM은_사설_IP면_http도_허용된다(config: Config) -> None:
    base = dict(
        EXTERNAL_LLM_ENABLED=True,
        EXTERNAL_LLM_MODEL_NAME="모델",
        EXTERNAL_LLM_API_KEY="키",
    )
    for url in (
        "http://172.16.1.100:30080/v1",
        "http://10.0.0.5:8000/v1",
        "https://192.168.0.10/v1",
    ):
        assert dataclasses.replace(config, EXTERNAL_LLM_BASE_URL=url, **base)
    # 공인 IP와 미등록 도메인은 http든 https든 거부
    for url in ("http://8.8.8.8/v1", "https://8.8.8.8/v1", "http://internal.example.com/v1"):
        with pytest.raises(ValueError, match="사설 IP"):
            dataclasses.replace(config, EXTERNAL_LLM_BASE_URL=url, **base)


def test_외부_LLM_설정이_비면_거부된다(config: Config) -> None:
    with pytest.raises(ValueError, match="EXTERNAL_LLM_API_KEY"):
        dataclasses.replace(
            config,
            EXTERNAL_LLM_ENABLED=True,
            EXTERNAL_LLM_BASE_URL="https://openrouter.ai/api/v1",
            EXTERNAL_LLM_MODEL_NAME="모델",
            EXTERNAL_LLM_API_KEY="",
        )


def test_설정은_불변이다(config: Config) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.AX_BASE_URL = "http://localhost:9999"  # type: ignore[misc]


def test_수치_설정_타입과_기본값(config: Config) -> None:
    assert isinstance(config.SEARCH_TOP_K, int)
    assert isinstance(config.RERANK_TOP_K, int)
    assert isinstance(config.RERANK_TOP_N, int)
    assert config.SEARCH_TOP_K > 0
    assert config.RERANK_TOP_N <= config.RERANK_TOP_K
    assert config.MAX_VERIFY_RETRY == 1
    assert config.HISTORY_MAX_TOKENS == 1500
    # 정확한 값은 .env가 덮어쓸 수 있으므로(개발 노트북 완화 등) 양수만 보장한다
    assert config.HTTP_TIMEOUT_SECONDS > 0
    # generate 전용 온도: 0(결정적)~1 범위만 보장 (.env로 조정 가능)
    assert 0.0 <= config.GENERATE_TEMPERATURE <= 1.0


def test_도메인_분류_체계() -> None:
    assert DOMAINS == ("HR", "TECH", "FINANCE_LEGAL", "GENERAL", "MANUAL", "DIRECTIVE")

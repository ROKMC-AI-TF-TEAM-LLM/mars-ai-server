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
    assert config.AX_BASE_URL.startswith("http://localhost:8000")
    assert config.EMBEDDING_SERVER_URL.startswith("http://localhost:8001")
    assert config.RERANKER_SERVER_URL.startswith("http://localhost:8002")


def test_localhost가_아닌_URL은_거부된다(config: Config) -> None:
    with pytest.raises(ValueError, match="에어갭"):
        dataclasses.replace(config, EMBEDDING_SERVER_URL="http://api.example.com/embed")


def test_설정은_불변이다(config: Config) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.AX_BASE_URL = "http://localhost:9999"  # type: ignore[misc]


def test_수치_설정_타입과_기본값(config: Config) -> None:
    assert isinstance(config.RERANK_TOP_K, int)
    assert isinstance(config.RERANK_TOP_N, int)
    assert config.RERANK_TOP_N <= config.RERANK_TOP_K
    assert config.MAX_VERIFY_RETRY == 1
    assert config.HISTORY_MAX_TOKENS == 1500
    assert config.HTTP_TIMEOUT_SECONDS == 60.0


def test_도메인_분류_체계() -> None:
    assert DOMAINS == ("HR", "TECH", "FINANCE_LEGAL", "GENERAL", "MANUAL", "DIRECTIVE")

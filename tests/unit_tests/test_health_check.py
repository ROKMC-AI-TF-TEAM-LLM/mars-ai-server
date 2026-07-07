"""shared/health.py 유닛 테스트 — HTTP/Milvus를 가짜로 대체해 집계 로직 검증."""

from __future__ import annotations

import pytest
import requests

from ax_rag.shared import health as health_module


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeMilvusClient:
    def __init__(self, exists: bool = True, exc: Exception | None = None) -> None:
        self._exists = exists
        self._exc = exc

    def has_collection(self, name: str) -> bool:
        if self._exc is not None:
            raise self._exc
        return self._exists


def _patch_milvus(monkeypatch: pytest.MonkeyPatch, client: _FakeMilvusClient) -> None:
    monkeypatch.setattr(health_module.vectorstore, "get_client", lambda: client)


def test_전부_정상이면_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_module.requests, "get", lambda url, timeout: _FakeResponse(200))
    _patch_milvus(monkeypatch, _FakeMilvusClient(exists=True))

    result = health_module.check_dependencies()
    assert result["status"] == "ok"
    assert set(result["services"]) == {"llm", "embedding", "reranker", "milvus"}
    assert all(service["ok"] for service in result["services"].values())


def test_임베딩_다운이면_degraded_해당_서비스만_실패(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> _FakeResponse:
        if ":8001" in url:
            raise requests.ConnectionError("connection refused")
        return _FakeResponse(200)

    monkeypatch.setattr(health_module.requests, "get", fake_get)
    _patch_milvus(monkeypatch, _FakeMilvusClient(exists=True))

    result = health_module.check_dependencies()
    assert result["status"] == "degraded"
    assert result["services"]["embedding"]["ok"] is False
    assert result["services"]["llm"]["ok"] is True
    assert result["services"]["reranker"]["ok"] is True


def test_HTTP_200이_아니면_실패로_판정(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> _FakeResponse:
        if ":8002" in url:
            return _FakeResponse(503)
        return _FakeResponse(200)

    monkeypatch.setattr(health_module.requests, "get", fake_get)
    _patch_milvus(monkeypatch, _FakeMilvusClient(exists=True))

    result = health_module.check_dependencies()
    assert result["status"] == "degraded"
    assert result["services"]["reranker"]["ok"] is False
    assert "503" in result["services"]["reranker"]["detail"]


def test_LLM은_OpenAI_호환_models_경로로_확인한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM/llama.cpp 어느 쪽이든 존재하는 공통 경로를 써야 한다."""
    called_urls: list[str] = []

    def fake_get(url: str, timeout: float) -> _FakeResponse:
        called_urls.append(url)
        return _FakeResponse(200)

    monkeypatch.setattr(health_module.requests, "get", fake_get)
    _patch_milvus(monkeypatch, _FakeMilvusClient(exists=True))

    health_module.check_dependencies()
    assert any(url.endswith("/models") for url in called_urls)


def test_Milvus_예외는_milvus만_실패로(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_module.requests, "get", lambda url, timeout: _FakeResponse(200))
    _patch_milvus(monkeypatch, _FakeMilvusClient(exc=RuntimeError("db 파일 잠김")))

    result = health_module.check_dependencies()
    assert result["status"] == "degraded"
    assert result["services"]["milvus"]["ok"] is False
    assert result["services"]["llm"]["ok"] is True


def test_컬렉션이_없어도_접속되면_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """첫 적재 전 상태는 장애가 아니다."""
    monkeypatch.setattr(health_module.requests, "get", lambda url, timeout: _FakeResponse(200))
    _patch_milvus(monkeypatch, _FakeMilvusClient(exists=False))

    result = health_module.check_dependencies()
    assert result["status"] == "ok"
    assert "첫 적재 전" in result["services"]["milvus"]["detail"]

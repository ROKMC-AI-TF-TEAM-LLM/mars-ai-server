"""serving 서버 API 계약 유닛 테스트.

실제 모델 없이 가짜 모델을 주입해 요청/응답 계약(차원, 점수 범위, 순서)과
로컬 경로 강제(Hub 폴백 금지)를 검증한다. 실제 모델 기동 검증은
tests/integration_tests/test_serving.py (integration 마커).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ax_rag.shared.config import get_config

_SERVING_DIR = Path(__file__).resolve().parents[2] / "serving"


def _load_serving_module(name: str) -> ModuleType:
    """serving/은 패키지가 아니므로 파일 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location(name, _SERVING_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embedding_server = _load_serving_module("embedding_server")
reranker_server = _load_serving_module("reranker_server")


class _FakeEmbedder:
    """BGEM3FlagModel.encode 계약을 흉내 내는 가짜 모델."""

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> dict[str, Any]:
        return {"dense_vecs": [[0.01] * embedding_server.EMBED_DIM for _ in texts]}


class _FakeReranker:
    """FlagReranker.compute_score 계약을 흉내 낸다 (쌍 1개면 float 반환)."""

    def compute_score(
        self, pairs: list[tuple[str, str]], normalize: bool, batch_size: int
    ) -> float | list[float]:
        scores = [0.9 - 0.1 * i for i in range(len(pairs))]
        return scores[0] if len(scores) == 1 else scores


def test_embed_응답은_텍스트당_1024차원_벡터() -> None:
    embedding_server.app.state.model = _FakeEmbedder()
    response = embedding_server.embed(
        embedding_server.EmbedRequest(texts=["연차는 며칠인가요?", "육아휴직 규정"])
    )
    assert len(response.embeddings) == 2
    assert all(len(vec) == 1024 for vec in response.embeddings)


def test_rerank_응답은_passages와_같은_순서의_0_1_점수() -> None:
    reranker_server.app.state.model = _FakeReranker()
    response = reranker_server.rerank(
        reranker_server.RerankRequest(
            query="육아휴직 기간", passages=["육아휴직은 1년", "법인카드 규정", "연차 규정"]
        )
    )
    assert len(response.scores) == 3
    assert all(0.0 <= s <= 1.0 for s in response.scores)
    assert response.scores == sorted(response.scores, reverse=True)  # 가짜 모델 순서 보존 확인


def test_rerank_passage_1개일_때_float_반환도_리스트로_감싼다() -> None:
    reranker_server.app.state.model = _FakeReranker()
    response = reranker_server.rerank(
        reranker_server.RerankRequest(query="질문", passages=["후보 하나"])
    )
    assert response.scores == [0.9]


@pytest.mark.parametrize("module", [embedding_server, reranker_server])
def test_모델_로컬_경로_없으면_즉시_실패한다(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """에어갭 규칙: 로컬 경로 부재 시 Hub 다운로드로 빠지지 않고 기동 실패해야 한다."""
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "./__no_such_model_dir__")
    monkeypatch.setenv("RERANKER_MODEL_PATH", "./__no_such_model_dir__")
    get_config.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="반입"):
            module._load_model()
    finally:
        get_config.cache_clear()

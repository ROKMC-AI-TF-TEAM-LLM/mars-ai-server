"""서빙 서버 통합 테스트 (실제 모델 기동 필요, 기본 skip).

실행 전제: embedding_server(8001), reranker_server(8002)가 떠 있어야 한다.
실행: make test-all
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest
import requests

from ax_rag.shared.config import get_config

pytestmark = pytest.mark.integration


def _base_url(service_url: str) -> str:
    """서비스 URL에서 스킴://호스트:포트만 추출한다."""
    parsed = urlparse(service_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def test_임베딩_서버_헬스체크() -> None:
    config = get_config()
    response = requests.get(f"{_base_url(config.EMBEDDING_SERVER_URL)}/health", timeout=10)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_리랭커_서버_헬스체크() -> None:
    config = get_config()
    response = requests.get(f"{_base_url(config.RERANKER_SERVER_URL)}/health", timeout=10)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed가_1024차원_벡터를_반환한다() -> None:
    config = get_config()
    response = requests.post(
        config.EMBEDDING_SERVER_URL,
        json={"texts": ["육아휴직은 얼마나 쓸 수 있나요?", "연차 이월 규정"]},
        timeout=config.HTTP_TIMEOUT_SECONDS,
    )
    assert response.status_code == 200
    embeddings = response.json()["embeddings"]
    assert len(embeddings) == 2
    assert all(len(vec) == 1024 for vec in embeddings)


def test_rerank가_0_1_점수를_같은_순서로_반환한다() -> None:
    config = get_config()
    passages = [
        "육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있다.",
        "법인카드 사용 한도는 부서별로 상이하다.",
    ]
    response = requests.post(
        config.RERANKER_SERVER_URL,
        json={"query": "육아휴직 기간은?", "passages": passages},
        timeout=config.HTTP_TIMEOUT_SECONDS,
    )
    assert response.status_code == 200
    scores = response.json()["scores"]
    assert len(scores) == len(passages)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # 관련 있는 첫 passage가 무관한 둘째보다 높아야 한다
    assert scores[0] > scores[1]

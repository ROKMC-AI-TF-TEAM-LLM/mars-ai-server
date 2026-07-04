"""bge-reranker-v2-m3 리랭커 서버 (포트 8002, interfaces.md §1·§5).

독립 FastAPI 프로세스. (질문, 후보 청크) 쌍의 관련도를 0~1로 채점한다.
모델은 반드시 오프라인 반입된 로컬 경로에서만 로드한다 (Hub 폴백 금지).

실행: python serving/reranker_server.py
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ax_rag.shared.config import get_config

# 에어갭 방어: FlagEmbedding은 _load_model에서 지연 임포트되므로,
# 그 전에 모듈 로드 시점에 오프라인 모드를 강제해 둔다
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("reranker_server")

PORT = 8002  # interfaces.md §1 고정 배정
BATCH_SIZE = 32  # RERANK_TOP_K=20이므로 통상 1배치로 처리된다
MAX_LENGTH = 512  # 질문 + 자식 청크(맥락 헤더 포함) 기준으로 충분한 길이


class RerankRequest(BaseModel):
    """POST /rerank 요청."""

    query: str
    passages: list[str] = Field(min_length=1)


class RerankResponse(BaseModel):
    """POST /rerank 응답. scores[i]는 passages[i]의 0~1 정규화 점수 (같은 순서)."""

    scores: list[float]


def _load_model() -> Any:
    """리랭커를 로컬 경로에서 로드한다. 경로가 없으면 즉시 실패한다 (Hub 폴백 금지)."""
    config = get_config()
    model_path = Path(config.RERANKER_MODEL_PATH)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"리랭커 모델 로컬 경로가 없다: {model_path} "
            "(에어갭 규칙상 Hub 다운로드 폴백은 없다. 모델을 먼저 반입할 것)"
        )
    # 지연 임포트: 서버 기동 시에만 필요하며, 유닛 테스트는 모델 없이 계약만 검증한다
    from FlagEmbedding import FlagReranker

    use_fp16 = config.RERANKER_DEVICE == "cuda"
    logger.info("리랭커 로드 시작: path=%s, device=%s", model_path, config.RERANKER_DEVICE)
    return FlagReranker(str(model_path), use_fp16=use_fp16, devices=config.RERANKER_DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """기동 시 모델을 1회 로드해 app.state에 보관한다."""
    app.state.model = _load_model()
    logger.info("리랭커 로드 완료. 포트 %d에서 대기", PORT)
    yield


app = FastAPI(title="bge-reranker-v2-m3 리랭커 서버", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """헬스체크."""
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    """(query, passage) 쌍마다 0~1 정규화 점수를 passages와 같은 순서로 반환한다.

    normalize=True로 sigmoid를 적용해 0~1 범위를 보장한다.
    동기 함수로 선언해 FastAPI가 스레드풀에서 실행하게 한다.
    """
    pairs = [(request.query, passage) for passage in request.passages]
    scores = app.state.model.compute_score(pairs, normalize=True, batch_size=BATCH_SIZE)
    # compute_score는 쌍이 1개면 float, 여러 개면 list를 반환한다
    if not isinstance(scores, list):
        scores = [scores]
    return RerankResponse(scores=[float(s) for s in scores])


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # 같은 서버의 LangGraph 앱만 호출하므로 루프백에만 바인딩한다 (에어갭)
    uvicorn.run(app, host="127.0.0.1", port=PORT)

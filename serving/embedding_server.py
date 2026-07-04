"""BGE-M3 임베딩 서버 (포트 8001, interfaces.md §1·§5).

독립 FastAPI 프로세스. LangGraph 앱이 HTTP로 호출하는 GPU 서비스이며,
모델은 반드시 오프라인 반입된 로컬 경로에서만 로드한다 (Hub 폴백 금지).

실행: python serving/embedding_server.py
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

logger = logging.getLogger("embedding_server")

PORT = 8001  # interfaces.md §1 고정 배정
EMBED_DIM = 1024  # BGE-M3 dense 차원
BATCH_SIZE = 32  # 배치 처리 단위 (대량 텍스트도 이 단위로 나눠 처리)
MAX_LENGTH = 512  # 자식 청크 150~200토큰 + 맥락 헤더 기준으로 충분한 길이


class EmbedRequest(BaseModel):
    """POST /embed 요청."""

    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    """POST /embed 응답. embeddings[i]는 texts[i]의 1024차원 dense 벡터."""

    embeddings: list[list[float]]


def _load_model() -> Any:
    """BGE-M3를 로컬 경로에서 로드한다. 경로가 없으면 즉시 실패한다 (Hub 폴백 금지)."""
    config = get_config()
    model_path = Path(config.EMBEDDING_MODEL_PATH)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"임베딩 모델 로컬 경로가 없다: {model_path} "
            "(에어갭 규칙상 Hub 다운로드 폴백은 없다. 모델을 먼저 반입할 것)"
        )
    # 지연 임포트: 서버 기동 시에만 필요하며, 유닛 테스트는 모델 없이 계약만 검증한다
    from FlagEmbedding import BGEM3FlagModel

    use_fp16 = config.EMBEDDING_DEVICE == "cuda"
    logger.info("BGE-M3 로드 시작: path=%s, device=%s", model_path, config.EMBEDDING_DEVICE)
    return BGEM3FlagModel(str(model_path), use_fp16=use_fp16, devices=config.EMBEDDING_DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """기동 시 모델을 1회 로드해 app.state에 보관한다."""
    app.state.model = _load_model()
    logger.info("BGE-M3 로드 완료. 포트 %d에서 대기", PORT)
    yield


app = FastAPI(title="BGE-M3 임베딩 서버", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """헬스체크."""
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """텍스트 목록을 1024차원 dense 벡터 목록으로 변환한다.

    동기 함수로 선언해 FastAPI가 스레드풀에서 실행하게 한다
    (GPU 추론이 이벤트 루프를 막지 않도록).
    """
    result = app.state.model.encode(request.texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    dense = result["dense_vecs"]
    return EmbedResponse(embeddings=[[float(x) for x in row] for row in dense])


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # 같은 서버의 LangGraph 앱만 호출하므로 루프백에만 바인딩한다 (에어갭)
    uvicorn.run(app, host="127.0.0.1", port=PORT)

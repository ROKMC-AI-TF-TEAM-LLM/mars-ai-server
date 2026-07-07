"""로컬 의존 서비스 헬스체크 (GET /health?deep=true의 재료).

MARS가 의존하는 로컬 서비스 4종(LLM, 임베딩, 리랭커, Milvus)의 도달 가능
여부를 집계한다. 질의가 실패하고 나서야 장애를 아는 것이 아니라, 미들웨어와
프론트의 서버 상태 화면에서 미리 볼 수 있게 하는 용도다.

에어갭 규칙: 검사 대상 URL은 전부 config에서 파생한다
(config가 localhost 강제를 보장).
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from ax_rag.shared import vectorstore
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 헬스체크 전용 timeout. 장애 상황에서도 응답이 빨리 돌아오도록
# 공통 timeout(60초)보다 짧게 잡는다
HEALTH_TIMEOUT_SECONDS = 5.0


def _service_base(url: str) -> str:
    """서비스 URL에서 scheme://host:port만 남긴다 (경로 제거)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _check_http(url: str) -> dict:
    """GET 1회로 도달 여부를 판정한다. {"ok": bool, "detail": str} 반환."""
    try:
        response = requests.get(url, timeout=HEALTH_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return {"ok": False, "detail": f"연결 실패 ({url}): {exc.__class__.__name__}"}
    if response.status_code == 200:
        return {"ok": True, "detail": f"HTTP 200 ({url})"}
    return {"ok": False, "detail": f"HTTP {response.status_code} ({url})"}


def _check_milvus() -> dict:
    """Milvus 접속 + 컬렉션 존재 확인. 컬렉션이 없어도 접속되면 ok (첫 적재 전 상태)."""
    config = get_config()
    try:
        exists = vectorstore.get_client().has_collection(config.MILVUS_COLLECTION)
    except Exception as exc:
        return {"ok": False, "detail": f"접속 실패: {exc.__class__.__name__}: {exc}"}
    detail = f"컬렉션 {config.MILVUS_COLLECTION} " + ("있음" if exists else "없음 (첫 적재 전)")
    return {"ok": True, "detail": detail}


def check_dependencies() -> dict:
    """로컬 서비스 4종 상태를 집계한다. 하나라도 실패면 status="degraded".

    LLM은 서빙(vLLM/llama.cpp)마다 전용 헬스 경로가 달라서, 둘 다 지원하는
    OpenAI 호환 GET {AX_BASE_URL}/models로 확인한다.
    """
    config = get_config()
    services = {
        "llm": _check_http(config.AX_BASE_URL.rstrip("/") + "/models"),
        "embedding": _check_http(_service_base(config.EMBEDDING_SERVER_URL) + "/health"),
        "reranker": _check_http(_service_base(config.RERANKER_SERVER_URL) + "/health"),
        "milvus": _check_milvus(),
    }
    down = [name for name, service in services.items() if not service["ok"]]
    if down:
        logger.warning("헬스체크 실패 서비스: %s", ", ".join(down))
    return {"status": "degraded" if down else "ok", "services": services}

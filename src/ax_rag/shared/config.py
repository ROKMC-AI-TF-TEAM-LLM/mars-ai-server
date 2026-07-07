"""전역 설정 모듈.

.env를 로드해 frozen dataclass로 노출한다. 설정 접근은 반드시
get_config()를 통해서만 하며, 다른 모듈에서 os.environ에 직접
접근하는 것을 금지한다 (환경변수를 읽는 곳은 이 파일이 유일하다).

에어갭 규칙: 모든 서비스 URL 기본값은 localhost이며, localhost가
아닌 호스트가 설정되면 기동 시점에 즉시 실패한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv

# 문서 도메인 체계 (Milvus domain 필드의 허용 값 = 요청 domain 필터의 허용 값)
# MANUAL=교범, DIRECTIVE=훈령 (도메인 한정 검색 모드용, interfaces.md §5)
DOMAINS: tuple[str, ...] = ("HR", "TECH", "FINANCE_LEGAL", "GENERAL", "MANUAL", "DIRECTIVE")

# 도메인 코드 → 화면 표시용 한글 라벨 (GET /capabilities로 프론트에 제공)
DOMAIN_LABELS: dict[str, str] = {
    "HR": "인사·복지",
    "TECH": "정보화·보안",
    "FINANCE_LEGAL": "재무·법무",
    "GENERAL": "일반",
    "MANUAL": "교범",
    "DIRECTIVE": "훈령",
}

# 한국어 토큰 수 근사: 문자수 / CHARS_PER_TOKEN (L40에서 실측 보정 예정)
CHARS_PER_TOKEN: float = 2.2

# 에어갭 규칙: 런타임 HTTP 호출이 허용되는 호스트
_ALLOWED_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1"})


@dataclass(frozen=True)
class Config:
    """전역 설정. 필드명은 .env 키와 1:1로 대응한다 (interfaces.md §8)."""

    # --- A.X 모델 서빙 (vLLM) ---
    AX_BASE_URL: str
    AX_MODEL_NAME: str
    AX_API_KEY: str

    # --- 임베딩 서버 (BGE-M3) ---
    EMBEDDING_SERVER_URL: str
    EMBEDDING_DEVICE: str
    EMBEDDING_MODEL_PATH: str  # 로컬 경로만 허용 (Hub ID 금지)

    # --- 리랭커 서버 (bge-reranker-v2-m3) ---
    RERANKER_SERVER_URL: str
    RERANKER_DEVICE: str
    RERANKER_MODEL_PATH: str  # 로컬 경로만 허용 (Hub ID 금지)
    RERANK_TOP_K: int
    RERANK_TOP_N: int

    # --- Milvus Lite (임베디드) ---
    MILVUS_LITE_PATH: str
    MILVUS_COLLECTION: str

    # --- BM25 키워드 검색 ---
    BM25_INDEX_PATH: str

    # --- 파이프라인 ---
    MAX_VERIFY_RETRY: int
    HISTORY_MAX_TOKENS: int

    # --- 감사 로그 ---
    AUDIT_LOG_PATH: str

    # --- 문서 업로드 저장 경로 (POST /documents가 받은 파일 원본 보관) ---
    UPLOAD_DIR: str = "./data/uploads"

    # --- 외부 서비스 호출 공통 timeout (초) ---
    HTTP_TIMEOUT_SECONDS: float = 60.0

    # --- 로그 레벨 (DEBUG/INFO/WARNING/ERROR) ---
    LOG_LEVEL: str = "INFO"

    # --- SSE text 이벤트 간 전송 간격(ms). 체감 스트리밍(타자기 효과)용 ---
    STREAM_TEXT_INTERVAL_MS: int = 200

    def __post_init__(self) -> None:
        """에어갭 검증: 서비스 URL이 localhost가 아니면 즉시 실패한다."""
        for name in ("AX_BASE_URL", "EMBEDDING_SERVER_URL", "RERANKER_SERVER_URL"):
            url: str = getattr(self, name)
            host = urlparse(url).hostname
            if host not in _ALLOWED_HOSTS:
                raise ValueError(
                    f"{name}에 localhost가 아닌 호스트는 허용되지 않는다 (에어갭 규칙): {url}"
                )


def _env_str(key: str, default: str) -> str:
    """환경변수 문자열 조회. 비어 있으면 기본값."""
    value = os.environ.get(key, "").strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    """환경변수 정수 조회. 비어 있으면 기본값."""
    value = os.environ.get(key, "").strip()
    return int(value) if value else default


def _env_float(key: str, default: float) -> float:
    """환경변수 실수 조회. 비어 있으면 기본값."""
    value = os.environ.get(key, "").strip()
    return float(value) if value else default


@lru_cache(maxsize=1)
def get_config() -> Config:
    """설정 싱글턴. 최초 호출 시 .env를 로드한다."""
    load_dotenv()
    return Config(
        AX_BASE_URL=_env_str("AX_BASE_URL", "http://localhost:8000/v1"),
        AX_MODEL_NAME=_env_str("AX_MODEL_NAME", "skt/A.X-4.0-Light"),
        AX_API_KEY=_env_str("AX_API_KEY", "EMPTY"),
        EMBEDDING_SERVER_URL=_env_str("EMBEDDING_SERVER_URL", "http://localhost:8001/embed"),
        EMBEDDING_DEVICE=_env_str("EMBEDDING_DEVICE", "cuda"),
        EMBEDDING_MODEL_PATH=_env_str("EMBEDDING_MODEL_PATH", "./models/bge-m3"),
        RERANKER_SERVER_URL=_env_str("RERANKER_SERVER_URL", "http://localhost:8002/rerank"),
        RERANKER_DEVICE=_env_str("RERANKER_DEVICE", "cuda"),
        RERANKER_MODEL_PATH=_env_str("RERANKER_MODEL_PATH", "./models/bge-reranker-v2-m3"),
        RERANK_TOP_K=_env_int("RERANK_TOP_K", 20),
        RERANK_TOP_N=_env_int("RERANK_TOP_N", 5),
        MILVUS_LITE_PATH=_env_str("MILVUS_LITE_PATH", "./data/milvus_ax.db"),
        MILVUS_COLLECTION=_env_str("MILVUS_COLLECTION", "company_docs"),
        BM25_INDEX_PATH=_env_str("BM25_INDEX_PATH", "./data/bm25_index"),
        MAX_VERIFY_RETRY=_env_int("MAX_VERIFY_RETRY", 1),
        HISTORY_MAX_TOKENS=_env_int("HISTORY_MAX_TOKENS", 1500),
        AUDIT_LOG_PATH=_env_str("AUDIT_LOG_PATH", "./data/audit_log.jsonl"),
        UPLOAD_DIR=_env_str("UPLOAD_DIR", "./data/uploads"),
        HTTP_TIMEOUT_SECONDS=_env_float("HTTP_TIMEOUT_SECONDS", 60.0),
        LOG_LEVEL=_env_str("LOG_LEVEL", "INFO"),
        STREAM_TEXT_INTERVAL_MS=_env_int("STREAM_TEXT_INTERVAL_MS", 200),
    )

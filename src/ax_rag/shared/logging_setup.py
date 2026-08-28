"""로깅 유틸: 통일 포맷 로거 팩토리.

- 각 모듈: ``logger = get_logger(__name__)``
- 진입점(main.py, serving/*, scripts/*): ``setup_logging()``을 함께 호출해
  서드파티(httpx, uvicorn 등) 로그까지 같은 포맷으로 맞춘다

출력 예: [19:26:31] INFO ax_rag.query_graph.nodes.router: 라우팅: domain=HR, ...
로그 레벨은 config.LOG_LEVEL(.env)로 제어한다.
"""

from __future__ import annotations

import logging

from ax_rag.shared.config import get_config

LOG_FORMAT = "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# milvus-lite 3.x가 구현하지 않은 gRPC 메서드 때문에 매 작업마다 나오는 소음.
# pymilvus는 분산 Milvus 서버를 전제로 AllocTimestamp(전역 타임스탬프 할당)를
# 호출하는데, 임베디드 milvus-lite에는 조율할 노드가 없어 구현이 없다.
# 클라이언트 타임스탬프로 폴백하며, **정합성에는 영향이 없다** — 적재 직후
# 전체 조회가 방금 insert를 보는지 실측으로 확인했다 (docs/milvus_lite_3x.md).
#
# 지우지 않으면 ERROR 트레이스백이 로그를 덮어 진짜 오류를 못 찾는다.
# 다만 **이 두 메시지만** 지운다 — 다른 gRPC 오류는 그대로 보여야 한다.
_MILVUS_LITE_NOISE = (
    ("grpc._server", "Method not implemented!"),
    ("pymilvus", "failed to get mvccTs"),
)


class MilvusLiteNoiseFilter(logging.Filter):
    """milvus-lite 3.x의 무해한 gRPC 미구현 소음만 걸러낸다."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            record.name.startswith(logger_name) and needle in message
            for logger_name, needle in _MILVUS_LITE_NOISE
        )


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """통일 포맷 로거를 반환한다. 모듈 상단에서 logger = get_logger(__name__)로 사용."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(get_config().LOG_LEVEL.upper())
        handler = logging.StreamHandler()
        handler.setFormatter(_make_formatter())
        logger.addHandler(handler)
        logger.propagate = False  # 루트 핸들러와의 중복 출력 방지
    return logger


def _attach_filter(target: logging.Logger | logging.Handler) -> None:
    """필터를 한 번만 붙인다 (중복 호출 안전)."""
    if not any(isinstance(f, MilvusLiteNoiseFilter) for f in target.filters):
        target.addFilter(MilvusLiteNoiseFilter())


def silence_milvus_lite_noise() -> None:
    """milvus-lite 소음 필터를 건다. 중복 호출 안전.

    ⚠️ 로거에 건 필터는 **그 로거가 직접 emit한 레코드에만** 적용된다.
    하위 로거(`pymilvus.orm.iterator`)에서 전파돼 온 레코드는 상위 로거의
    필터를 거치지 않고 핸들러로 바로 간다. 그래서 로거와 핸들러 **양쪽에**
    건다 — 로거만 걸면 전파돼 오는 소음이 그대로 남는다 (실측 확인).
    """
    for logger_name, _ in _MILVUS_LITE_NOISE:
        logger = logging.getLogger(logger_name)
        _attach_filter(logger)
        for handler in logger.handlers:
            _attach_filter(handler)
    for handler in logging.getLogger().handlers:
        _attach_filter(handler)


def setup_logging(level: str | None = None) -> None:
    """루트 로거에 같은 포맷을 적용한다 (서드파티 로그용). 중복 호출 안전."""
    resolved = (level or get_config().LOG_LEVEL).upper()
    silence_milvus_lite_noise()
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(_make_formatter())
        root.setLevel(resolved)
        return
    logging.basicConfig(level=resolved, format=LOG_FORMAT, datefmt=DATE_FORMAT)

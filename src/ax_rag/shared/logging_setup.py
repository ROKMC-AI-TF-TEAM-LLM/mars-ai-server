"""로깅 유틸: 통일 포맷 로거 팩토리.

- 각 모듈: ``logger = get_logger(__name__)``
- 진입점(main.py, serving/*, scripts/*): ``setup_logging()``을 함께 호출해
  서드파티(httpx, uvicorn 등) 로그까지 같은 포맷으로 맞춘다

출력 예: [19:26:31] INFO ax_rag.retrieval_graph.nodes.router: 라우팅: domain=HR, ...
로그 레벨은 config.LOG_LEVEL(.env)로 제어한다.
"""

from __future__ import annotations

import logging

from ax_rag.shared.config import get_config

# 밀리초 포함: 스트리밍 조각 간격(수십 ms)이 로그에서 구분돼야 한다
LOG_FORMAT = "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


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


def setup_logging(level: str | None = None) -> None:
    """루트 로거에 같은 포맷을 적용한다 (서드파티 로그용). 중복 호출 안전."""
    resolved = (level or get_config().LOG_LEVEL).upper()
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(_make_formatter())
        root.setLevel(resolved)
        return
    logging.basicConfig(level=resolved, format=LOG_FORMAT, datefmt=DATE_FORMAT)

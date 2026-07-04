"""로깅 형식 통일: 시각 | 레벨 | 모듈명 | 메시지.

모든 진입점(main.py, serving/*, scripts/*)이 이 모듈의 setup_logging()을
호출해 동일한 포맷을 쓴다. print 금지 규칙(CLAUDE.md)과 함께 사용.

예: 2026-07-04 19:20:31 | INFO | ax_rag.retrieval_graph.nodes.router | 라우팅: domain=HR, ...
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """루트 로거에 통일 포맷을 적용한다. 중복 호출해도 안전하다."""
    root = logging.getLogger()
    if root.handlers:
        # 이미 핸들러가 있으면(예: 재호출) 포맷만 맞춘다
        for handler in root.handlers:
            handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)

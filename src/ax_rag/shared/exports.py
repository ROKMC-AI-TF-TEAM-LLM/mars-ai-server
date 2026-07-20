"""생성 문서(EXPORT_DIR) 임시 보관소 정리.

EXPORT_DIR은 도구가 만든 파일(HWPX 등)의 임시 보관소다 — 미들웨어가
SSE file 이벤트를 신호로 즉시 가져가 자기 저장소에 보관하므로(interfaces.md
§5), 여기 파일은 TTL(EXPORT_TTL_HOURS, 기본 24시간)만 지나면 지워도 된다.

정리는 **새 파일을 생성하는 시점에 기회적으로** 수행한다: 디렉터리가
커지는 유일한 경로가 생성이므로 생성 시 정리만으로 크기가 유한하게
유지된다. 별도 스케줄러·백그라운드 스레드가 필요 없어 단일 워커 전제
(CLAUDE.md)와도 정합한다.
"""

from __future__ import annotations

import time
from pathlib import Path

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)


def cleanup_expired_exports() -> int:
    """EXPORT_DIR에서 TTL이 지난 파일을 삭제한다. 삭제 건수 반환.

    TTL이 0 이하면 아무것도 하지 않는다 (정리 비활성).
    삭제 실패(파일 잠김 등)는 경고만 남기고 다음 기회에 다시 시도된다.
    """
    config = get_config()
    if config.EXPORT_TTL_HOURS <= 0:
        return 0
    export_dir = Path(config.EXPORT_DIR)
    if not export_dir.is_dir():
        return 0

    cutoff = time.time() - config.EXPORT_TTL_HOURS * 3600
    deleted = 0
    for path in export_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("만료 산출물 삭제 실패 (다음 정리 때 재시도): %s", path.name)
    if deleted:
        logger.info("만료 산출물 정리: %d건 삭제 (TTL %d시간)", deleted, config.EXPORT_TTL_HOURS)
    return deleted

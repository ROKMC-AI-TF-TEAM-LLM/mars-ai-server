"""단일 워커의 검색/저장 교체 동기화 및 미완료 갱신 감지."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ax_rag.shared.config import get_config


class StoreGate:
    """검색끼리는 병렬 실행하고, 저장소 교체는 검색 전체와 배타적으로 실행한다."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer: int | None = None
        self._depth = 0
        self._waiting_writers = 0

    def acquire(self) -> None:
        identity = threading.get_ident()
        with self._condition:
            if self._writer == identity:
                self._depth += 1
                return
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers:
                    self._condition.wait()
                self._writer = identity
                self._depth = 1
            finally:
                self._waiting_writers -= 1

    def release(self) -> None:
        with self._condition:
            if self._writer != threading.get_ident():
                raise RuntimeError("저장소 잠금 소유자가 아니다")
            self._depth -= 1
            if not self._depth:
                self._writer = None
                self._condition.notify_all()

    def __enter__(self) -> "StoreGate":
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._condition:
            while self._writer is not None or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()


STORE_LOCK = StoreGate()


def recovery_path() -> Path:
    """BM25와 함께 보관하는 갱신 복구 저널. 재시작 후에도 남는다."""
    return Path(get_config().BM25_INDEX_PATH) / "ingest-recovery.json"


def ensure_store_ready() -> None:
    """저널이 남은 경우 복구 전 검색·추가 변경을 거부한다 (fail-closed)."""
    if recovery_path().exists():
        raise RuntimeError("미완료 문서 갱신이 있다. recover_ingest.py로 복구한 뒤 재시도할 것")

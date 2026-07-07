"""적재 작업(job) 인메모리 레지스트리 (POST /documents 백그라운드 작업 추적).

단일 uvicorn 워커 전제라 프로세스 메모리로 충분하다. 서버가 재시작되면
작업 이력은 사라진다 — 적재 결과(청크)는 Milvus에 남으므로 이력만의 손실이다.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# 상태 전이: queued → running → done | error
JOB_STATUSES: tuple[str, ...] = ("queued", "running", "done", "error")


def _iso(timestamp: float | None) -> str | None:
    """unix timestamp → ISO 문자열 (초 단위). None은 그대로."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


@dataclass
class IngestJob:
    """적재 작업 1건의 상태."""

    job_id: str
    source_doc: str
    domain: str
    owning_department: str
    visibility: str
    status: str = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    chunks_indexed: int | None = None  # done일 때 적재된 자식 청크 수
    deleted_chunks: int | None = None  # 갱신 적재로 삭제된 기존 자식 청크 수
    error: str | None = None  # error일 때 사유

    @property
    def finished(self) -> bool:
        """종결(성공/실패) 여부."""
        return self.status in ("done", "error")

    def to_dict(self) -> dict:
        """API 응답용 직렬화 (시각은 ISO 문자열)."""
        return {
            "job_id": self.job_id,
            "source_doc": self.source_doc,
            "domain": self.domain,
            "owning_department": self.owning_department,
            "visibility": self.visibility,
            "status": self.status,
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "chunks_indexed": self.chunks_indexed,
            "deleted_chunks": self.deleted_chunks,
            "error": self.error,
        }


class IngestJobRegistry:
    """스레드 안전한 작업 레지스트리.

    dict의 삽입 순서 = 제출 순서를 유지하고, 상한 초과 시 종결된 작업 중
    오래된 것부터 정리한다 (진행 중인 작업은 정리하지 않는다).
    """

    def __init__(self, max_jobs: int = 100) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(
        self, source_doc: str, domain: str, owning_department: str, visibility: str
    ) -> IngestJob:
        """새 작업을 queued 상태로 등록한다."""
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            source_doc=source_doc,
            domain=domain,
            owning_department=owning_department,
            visibility=visibility,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._prune()
        return job

    def get(self, job_id: str) -> IngestJob | None:
        """작업 조회. 없으면 None (정리됐거나 서버 재시작으로 소실)."""
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[IngestJob]:
        """최근 제출 순(최신 먼저) 작업 목록."""
        with self._lock:
            return list(reversed(list(self._jobs.values())))[:limit]

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "running"
                job.started_at = time.time()

    def mark_done(self, job_id: str, chunks_indexed: int, deleted_chunks: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "done"
                job.finished_at = time.time()
                job.chunks_indexed = chunks_indexed
                job.deleted_chunks = deleted_chunks

    def mark_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.finished_at = time.time()
                job.error = error

    def _prune(self) -> None:
        """상한 초과분을 종결된 작업 중 오래된 것부터 제거한다 (_lock 보유 상태에서 호출)."""
        while len(self._jobs) > self._max_jobs:
            oldest_finished = next(
                (job_id for job_id, job in self._jobs.items() if job.finished), None
            )
            if oldest_finished is None:
                return  # 전부 진행 중이면 정리하지 않는다 (상한 일시 초과 허용)
            del self._jobs[oldest_finished]

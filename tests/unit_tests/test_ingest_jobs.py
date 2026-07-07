"""shared/ingest_jobs.py 유닛 테스트 — 상태 전이, 최신순 조회, 정리(prune) 규칙."""

from __future__ import annotations

from ax_rag.shared.ingest_jobs import IngestJobRegistry


def _registry(max_jobs: int = 100) -> IngestJobRegistry:
    return IngestJobRegistry(max_jobs=max_jobs)


def test_생성_직후는_queued_상태() -> None:
    registry = _registry()
    job = registry.create("휴가규정.md", "HR", "HR_TEAM", "ALL")
    assert job.status == "queued"
    assert job.started_at is None
    assert registry.get(job.job_id) is job


def test_상태_전이_running_done() -> None:
    registry = _registry()
    job = registry.create("훈령.pdf", "DIRECTIVE", "HQ", "ALL")

    registry.mark_running(job.job_id)
    assert job.status == "running"
    assert job.started_at is not None

    registry.mark_done(job.job_id, chunks_indexed=42, deleted_chunks=10)
    assert job.status == "done"
    assert job.finished_at is not None
    assert job.chunks_indexed == 42
    assert job.deleted_chunks == 10  # 갱신 적재로 지운 기존 청크
    assert job.error is None


def test_상태_전이_error() -> None:
    registry = _registry()
    job = registry.create("스캔본.pdf", "HR", "HR_TEAM", "ALL")
    registry.mark_running(job.job_id)
    registry.mark_error(job.job_id, "ValueError: 텍스트를 추출하지 못했다")
    assert job.status == "error"
    assert "추출하지 못했다" in (job.error or "")
    assert job.finished_at is not None


def test_미지의_job_id는_None() -> None:
    assert _registry().get("없는아이디") is None


def test_recent는_최신_제출_순() -> None:
    registry = _registry()
    first = registry.create("문서1.md", "HR", "HR_TEAM", "ALL")
    second = registry.create("문서2.md", "HR", "HR_TEAM", "ALL")
    third = registry.create("문서3.md", "HR", "HR_TEAM", "ALL")

    recent = registry.recent(limit=2)
    assert [job.job_id for job in recent] == [third.job_id, second.job_id]
    assert len(registry.recent(limit=10)) == 3
    assert registry.recent(limit=10)[-1].job_id == first.job_id


def test_상한_초과_시_종결된_작업부터_정리한다() -> None:
    registry = _registry(max_jobs=2)
    done_job = registry.create("완료된문서.md", "HR", "HR_TEAM", "ALL")
    registry.mark_done(done_job.job_id, chunks_indexed=1, deleted_chunks=0)
    running_job = registry.create("적재중문서.md", "HR", "HR_TEAM", "ALL")
    registry.mark_running(running_job.job_id)

    newest = registry.create("새문서.md", "HR", "HR_TEAM", "ALL")

    assert registry.get(done_job.job_id) is None  # 종결된 가장 오래된 것이 정리됨
    assert registry.get(running_job.job_id) is not None  # 진행 중은 보존
    assert registry.get(newest.job_id) is not None


def test_전부_진행_중이면_정리하지_않는다() -> None:
    registry = _registry(max_jobs=1)
    first = registry.create("문서1.md", "HR", "HR_TEAM", "ALL")
    second = registry.create("문서2.md", "HR", "HR_TEAM", "ALL")
    # 둘 다 미종결 → 상한을 일시 초과해도 잃지 않는다
    assert registry.get(first.job_id) is not None
    assert registry.get(second.job_id) is not None


def test_to_dict는_ISO_시각_문자열을_쓴다() -> None:
    registry = _registry()
    job = registry.create("휴가규정.md", "HR", "HR_TEAM", "ALL")
    payload = job.to_dict()
    assert payload["status"] == "queued"
    assert isinstance(payload["submitted_at"], str)
    assert "T" in payload["submitted_at"]  # ISO 형식
    assert payload["started_at"] is None
    assert payload["chunks_indexed"] is None

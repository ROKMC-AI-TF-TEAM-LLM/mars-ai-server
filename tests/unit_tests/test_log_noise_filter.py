"""milvus-lite 3.x의 무해한 gRPC 소음만 걸러낸다.

pymilvus는 분산 Milvus 서버를 전제로 AllocTimestamp를 호출하는데 임베디드
milvus-lite에는 구현이 없다. 매 작업마다 ERROR 트레이스백이 쌓여 진짜 오류를
덮으므로 지운다 — 정합성 영향이 없음은 실측으로 확인했다
(docs/milvus_lite_3x.md).

★ 다른 gRPC 오류까지 지우면 안 된다. 그게 이 테스트의 핵심이다.
"""

from __future__ import annotations

import logging

import pytest

from ax_rag.shared.logging_setup import MilvusLiteNoiseFilter, silence_milvus_lite_noise


def _record(name: str, message: str, level: int = logging.ERROR) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("grpc._server", "Exception calling application: Method not implemented!"),
        ("pymilvus", "failed to get mvccTs from milvus server, use client-side ts instead"),
        ("pymilvus.client", "failed to get mvccTs from milvus server"),
    ],
)
def test_알려진_소음은_걸러낸다(name: str, message: str) -> None:
    assert MilvusLiteNoiseFilter().filter(_record(name, message)) is False


@pytest.mark.parametrize(
    ("name", "message"),
    [
        # ★ 진짜 gRPC 오류 — 이게 걸러지면 장애를 놓친다
        ("grpc._server", "Exception calling application: Connection refused"),
        ("grpc._server", "Deadline exceeded"),
        ("pymilvus", "collection not found"),
        ("ax_rag.shared.vectorstore", "Method not implemented!"),  # 다른 로거는 대상 아님
    ],
)
def test_진짜_오류는_통과시킨다(name: str, message: str) -> None:
    assert MilvusLiteNoiseFilter().filter(_record(name, message)) is True


def test_하위_로거에서_전파된_소음도_걸러낸다() -> None:
    """★ 회귀 방지 — 실제로 놓쳤던 경우.

    로거에 건 필터는 **그 로거가 직접 emit한 레코드에만** 적용된다.
    mvccTs 경고는 pymilvus.orm.iterator가 내고 pymilvus 핸들러로 전파되므로,
    pymilvus 로거에만 필터를 걸면 그대로 통과한다.
    """
    parent = logging.getLogger("pymilvus")
    child = logging.getLogger("pymilvus.orm.iterator")

    captured: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Collector()
    parent.addHandler(handler)
    parent.setLevel(logging.WARNING)
    try:
        silence_milvus_lite_noise()  # 핸들러가 붙은 뒤에 호출해야 한다
        child.warning("failed to get mvccTs from milvus server, use client-side ts instead")
        child.warning("collection not found")
    finally:
        parent.removeHandler(handler)

    messages = [r.getMessage() for r in captured]
    assert not any("mvccTs" in m for m in messages)
    assert any("collection not found" in m for m in messages)


def test_필터를_두_번_걸어도_하나만_붙는다() -> None:
    """get_client와 setup_logging 양쪽에서 불린다 — 중복 등록되면 안 된다."""
    logger = logging.getLogger("grpc._server")
    before = [f for f in logger.filters if isinstance(f, MilvusLiteNoiseFilter)]
    for f in before:
        logger.removeFilter(f)

    silence_milvus_lite_noise()
    silence_milvus_lite_noise()

    attached = [f for f in logger.filters if isinstance(f, MilvusLiteNoiseFilter)]
    assert len(attached) == 1


def test_실제_로거에_걸면_소음이_기록되지_않는다() -> None:
    silence_milvus_lite_noise()
    logger = logging.getLogger("grpc._server")

    captured: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Collector()
    logger.addHandler(handler)
    try:
        logger.error("Exception calling application: Method not implemented!")
        logger.error("Exception calling application: Connection refused")
    finally:
        logger.removeHandler(handler)

    messages = [r.getMessage() for r in captured]
    assert not any("Method not implemented" in m for m in messages)
    assert any("Connection refused" in m for m in messages)

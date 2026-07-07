"""문서 적재/삭제 공용 로직 (API `POST/DELETE /documents`와 스크립트가 함께 사용).

동시 실행 제어: 적재와 삭제는 BM25 인덱스 전체 재빌드를 포함하므로 둘이
동시에 돌면 인덱스가 꼬인다. 모듈 잠금(_LOCK)으로 한 번에 하나만 실행한다
(단일 uvicorn 워커 전제이므로 프로세스 내 잠금으로 충분, CLAUDE.md).
"""

from __future__ import annotations

import threading
from pathlib import Path

from ax_rag.indexer_graph.graph import graph, rebuild_bm25
from ax_rag.indexer_graph.loaders import load_document
from ax_rag.shared import parent_store, vectorstore
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 적재·삭제 직렬화 잠금 (BM25 전체 재빌드의 동시 실행 방지)
_LOCK = threading.Lock()


class IngestBusyError(RuntimeError):
    """다른 적재/삭제 작업이 진행 중이라 잠금을 얻지 못했다."""


def ingest_file(path: Path, domain: str, owning_department: str, visibility: str) -> dict:
    """문서 1건 적재(갱신): 텍스트 추출 검증 → 기존 청크 삭제 → 재적재.

    같은 파일명이 이미 적재돼 있으면 갱신(삭제 후 재적재)이다.
    텍스트 추출을 삭제보다 먼저 수행한다 — 추출 실패(스캔본 PDF 등) 시
    기존 데이터를 지우지 않기 위해서다.

    반환: {"source_doc", "deleted_children", "deleted_parents", "chunks_indexed"}
    추출 실패 시 ValueError.
    """
    source_doc = path.name
    text, sections = load_document(path)
    if not text.strip():
        raise ValueError(f"{source_doc}: 텍스트를 추출하지 못했다 (스캔본 PDF?)")

    with _LOCK:
        deleted_children = vectorstore.delete_by_source_doc(source_doc)
        deleted_parents = parent_store.delete_by_source_doc(source_doc)
        if deleted_children or deleted_parents:
            logger.info(
                "%s: 기존 자식 %d건·부모 %d건 삭제 (갱신 적재)",
                source_doc,
                deleted_children,
                deleted_parents,
            )
        result = graph.invoke(
            {
                "text": text,
                "source_doc": source_doc,
                "domain": domain,
                "owning_department": owning_department,
                "visibility": visibility,
                "sections": sections,
            }
        )
    return {
        "source_doc": source_doc,
        "deleted_children": deleted_children,
        "deleted_parents": deleted_parents,
        "chunks_indexed": int(result.get("chunks_indexed") or 0),
    }


def delete_document(source_doc: str, wait_seconds: float = 10.0) -> dict:
    """문서 1건 삭제: 자식·부모 청크 삭제 후 BM25 전체 재빌드.

    진행 중인 적재/삭제가 있으면 wait_seconds까지 기다렸다가 IngestBusyError.
    삭제된 청크가 없으면(미적재 문서) BM25 재빌드를 건너뛴다 — BM25는 부분
    삭제가 불가하므로, 삭제가 실제로 일어났을 때만 전체 재빌드로 반영한다.

    반환: {"source_doc", "deleted_children", "deleted_parents"}
    """
    if not _LOCK.acquire(timeout=wait_seconds):
        raise IngestBusyError("다른 적재/삭제 작업이 진행 중이다")
    try:
        deleted_children = vectorstore.delete_by_source_doc(source_doc)
        deleted_parents = parent_store.delete_by_source_doc(source_doc)
        if deleted_children or deleted_parents:
            rebuild_bm25()
        logger.info(
            "문서 삭제: %s → 자식 %d건·부모 %d건 (BM25 재빌드=%s)",
            source_doc,
            deleted_children,
            deleted_parents,
            bool(deleted_children or deleted_parents),
        )
        return {
            "source_doc": source_doc,
            "deleted_children": deleted_children,
            "deleted_parents": deleted_parents,
        }
    finally:
        _LOCK.release()

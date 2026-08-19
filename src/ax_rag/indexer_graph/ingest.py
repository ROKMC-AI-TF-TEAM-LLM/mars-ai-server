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


def _purge_document(source_doc: str, project_id: str) -> tuple[int, int]:
    """문서 1건의 자식·부모 청크를 지운다. (자식 건수, 부모 건수) 반환.

    ★ 순서가 중요하다: 부모는 이름이 아니라 **자식이 참조하던 parent_id**로 지우므로,
    자식을 지우기 **전에** 그 목록을 먼저 모아야 한다. 이름으로 지우면 같은 파일명을
    쓰는 다른 프로젝트의 부모까지 사라진다.

    잠금은 호출부가 잡는다 (적재·삭제 경로가 각각 다른 범위로 잡기 때문).
    """
    parent_ids = vectorstore.fetch_parent_ids(source_doc, project_id)
    deleted_children = vectorstore.delete_by_source_doc(source_doc, project_id)
    deleted_parents = parent_store.delete_by_parent_ids(parent_ids)
    return deleted_children, deleted_parents


def ingest_file(
    path: Path,
    domain: str,
    owning_department: str,
    visibility: str,
    project_id: str = "",
) -> dict:
    """문서 1건 적재(갱신): 텍스트 추출 검증 → 기존 청크 삭제 → 재적재.

    **같은 (project_id, source_doc)** 이 이미 적재돼 있으면 갱신이다. 파일명이
    같아도 프로젝트가 다르면 **별개 문서**라 서로 덮어쓰지 않는다.
    텍스트 추출을 삭제보다 먼저 수행한다 — 추출 실패(스캔본 PDF 등) 시
    기존 데이터를 지우지 않기 위해서다.

    project_id는 ""(전사 공용)가 기본이다.

    반환: {"source_doc", "project_id", "deleted_children", "deleted_parents",
          "chunks_indexed"}. 추출 실패 시 ValueError.
    """
    source_doc = path.name
    text, sections = load_document(path)
    if not text.strip():
        raise ValueError(f"{source_doc}: 텍스트를 추출하지 못했다 (스캔본 PDF?)")

    with _LOCK:
        deleted_children, deleted_parents = _purge_document(source_doc, project_id)
        if deleted_children or deleted_parents:
            logger.info(
                "%s(project=%s): 기존 자식 %d건·부모 %d건 삭제 (갱신 적재)",
                source_doc,
                project_id or "전사",
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
                "project_id": project_id,
                "sections": sections,
            }
        )
    return {
        "source_doc": source_doc,
        "project_id": project_id,
        "deleted_children": deleted_children,
        "deleted_parents": deleted_parents,
        "chunks_indexed": int(result.get("chunks_indexed") or 0),
    }


def delete_project(project_id: str, wait_seconds: float = 10.0) -> dict:
    """프로젝트에 속한 문서를 전부 삭제한다 (자식·부모 + BM25 1회 재빌드).

    부모는 이름이 아니라 **자식이 참조하던 parent_id**로 지운다 — 같은 파일명을
    쓰는 전사·타 프로젝트 문서의 부모를 건드리지 않기 위해서다.
    BM25는 문서마다 재빌드하지 않고 전부 지운 뒤 한 번만 돌린다.

    반환: {"project_id", "documents", "deleted_children", "deleted_parents"}
    """
    if not project_id:
        raise ValueError("project_id가 비어 있다 (전사 문서 전체 삭제 방지)")
    if not _LOCK.acquire(timeout=wait_seconds):
        raise IngestBusyError("다른 적재/삭제 작업이 진행 중이다")
    try:
        documents = [
            doc["source_doc"]
            for doc in vectorstore.list_documents()
            if doc.get("project_id") == project_id
        ]
        if not documents:
            return {
                "project_id": project_id,
                "documents": [],
                "deleted_children": 0,
                "deleted_parents": 0,
            }

        # 자식을 지우기 전에 부모 참조를 모은다 (문서마다 project 스코프로)
        parent_ids: list[str] = []
        for name in documents:
            parent_ids.extend(vectorstore.fetch_parent_ids(name, project_id))
        deleted_children = vectorstore.delete_by_project(project_id)
        deleted_parents = parent_store.delete_by_parent_ids(parent_ids)
        rebuild_bm25()
        logger.info(
            "프로젝트 삭제: %s → 문서 %d건, 자식 %d건·부모 %d건",
            project_id,
            len(documents),
            deleted_children,
            deleted_parents,
        )
        return {
            "project_id": project_id,
            "documents": documents,
            "deleted_children": deleted_children,
            "deleted_parents": deleted_parents,
        }
    finally:
        _LOCK.release()


def delete_document(source_doc: str, project_id: str = "", wait_seconds: float = 10.0) -> dict:
    """문서 1건 삭제: 자식·부모 청크 삭제 후 BM25 전체 재빌드.

    **(project_id, source_doc) 복합키로 지운다.** project_id를 생략하면 전사 공용
    문서만 대상이며, 같은 파일명의 프로젝트 문서는 건드리지 않는다.

    진행 중인 적재/삭제가 있으면 wait_seconds까지 기다렸다가 IngestBusyError.
    삭제된 청크가 없으면(미적재 문서) BM25 재빌드를 건너뛴다 — BM25는 부분
    삭제가 불가하므로, 삭제가 실제로 일어났을 때만 전체 재빌드로 반영한다.

    반환: {"source_doc", "project_id", "deleted_children", "deleted_parents"}
    """
    if not _LOCK.acquire(timeout=wait_seconds):
        raise IngestBusyError("다른 적재/삭제 작업이 진행 중이다")
    try:
        deleted_children, deleted_parents = _purge_document(source_doc, project_id)
        if deleted_children or deleted_parents:
            rebuild_bm25()
        logger.info(
            "문서 삭제: %s(project=%s) → 자식 %d건·부모 %d건 (BM25 재빌드=%s)",
            source_doc,
            project_id or "전사",
            deleted_children,
            deleted_parents,
            bool(deleted_children or deleted_parents),
        )
        return {
            "source_doc": source_doc,
            "project_id": project_id,
            "deleted_children": deleted_children,
            "deleted_parents": deleted_parents,
        }
    finally:
        _LOCK.release()

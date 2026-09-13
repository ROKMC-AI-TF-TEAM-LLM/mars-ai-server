"""문서 적재/삭제 공용 로직 (API `POST/DELETE /documents`와 스크립트가 함께 사용).

동시 실행 제어: 적재와 삭제는 BM25 인덱스 전체 재빌드를 포함하므로 둘이
동시에 돌면 인덱스가 꼬인다. 모듈 잠금(_LOCK)으로 한 번에 하나만 실행한다
(단일 uvicorn 워커 전제이므로 프로세스 내 잠금으로 충분, CLAUDE.md).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from ax_rag.indexer_graph.graph import chunk, prepare_rows, rebuild_bm25
from ax_rag.indexer_graph.loaders import load_document
from ax_rag.shared import parent_store, vectorstore
from ax_rag.shared.config import get_config
from ax_rag.shared.departments import validate_department
from ax_rag.shared.logging_setup import get_logger
from ax_rag.shared.store_guard import STORE_LOCK, ensure_store_ready, recovery_path

logger = get_logger(__name__)

# 적재·삭제 직렬화 잠금 (BM25 전체 재빌드의 동시 실행 방지)
_LOCK = threading.Lock()


class IngestBusyError(RuntimeError):
    """다른 적재/삭제 작업이 진행 중이라 잠금을 얻지 못했다."""


def _restore_snapshot(snapshot: dict) -> None:
    """갱신 전 행으로 되돌린다. 반복 실행 가능하며 실패 시 저널을 유지한다."""
    config = get_config()
    if (snapshot["milvus_uri"], snapshot["collection"]) != (
        config.MILVUS_LITE_PATH,
        config.MILVUS_COLLECTION,
    ):
        raise ValueError("복구 저널의 Milvus 설정이 현재 설정과 다르다")
    vectorstore.delete_by_source_doc(snapshot["source_doc"], snapshot["project_id"])
    parent_store.delete_by_parent_ids(
        [p["parent_id"] for p in snapshot["parents"]] + snapshot["new_parent_ids"]
    )
    if parent_store.insert_parents(snapshot["parents"]) != len(snapshot["parents"]):
        raise RuntimeError("기존 부모 청크 복구 건수가 일치하지 않는다")
    restored = vectorstore.insert_children(snapshot["children"])
    if restored != len(snapshot["children"]):
        raise RuntimeError("기존 자식 청크 복구 건수가 일치하지 않는다")
    rebuild_bm25()
    recovery_path().unlink()


def recover_pending_ingest() -> bool:
    """서비스를 중지한 운영자가 미완료 갱신을 복구한다. 저널이 없으면 False."""
    with _LOCK, STORE_LOCK:
        path = recovery_path()
        if not path.exists():
            return False
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _restore_snapshot(snapshot)
        logger.info("미완료 갱신 복구 완료: %s", snapshot["source_doc"])
        return True


def _replace_document(state: dict, rows: list[dict]) -> tuple[int, int]:
    """준비된 문서를 교체한다. 예외 시 복구하며 실패 저널은 검색을 차단한다."""
    with STORE_LOCK:
        ensure_store_ready()
        source_doc, project_id = state["source_doc"], state["project_id"]
        old_rows = vectorstore.fetch_document_children(source_doc, project_id)
        old_parents: list[dict] = []
        for parent_id in sorted({row["parent_id"] for row in old_rows}):
            text = parent_store.get_parent(parent_id)
            if not text:
                raise RuntimeError(f"백업할 부모 청크가 없다: {parent_id}")
            old_parents.append(
                {"parent_id": parent_id, "parent_text": text, "source_doc": source_doc}
            )
        config = get_config()
        snapshot = {
            "source_doc": source_doc,
            "project_id": project_id,
            "milvus_uri": config.MILVUS_LITE_PATH,
            "collection": config.MILVUS_COLLECTION,
            "children": old_rows,
            "parents": old_parents,
            "new_parent_ids": [p["parent_id"] for p in state["parents"]],
        }
        # 저널을 디스크에 확정하기 전에는 기존 데이터를 변경하지 않는다.
        path = recovery_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as journal:
            json.dump(snapshot, journal, ensure_ascii=False, default=float)
            journal.flush()
            os.fsync(journal.fileno())
        temporary.replace(path)
        try:
            deleted_children = vectorstore.delete_by_source_doc(source_doc, project_id)
            if deleted_children != len(old_rows):
                raise RuntimeError("기존 자식 청크 삭제 건수가 일치하지 않는다")
            deleted_parents = parent_store.delete_by_parent_ids(
                [p["parent_id"] for p in old_parents]
            )
            if parent_store.insert_parents(state["parents"]) != len(state["parents"]):
                raise RuntimeError("새 부모 청크 적재 건수가 일치하지 않는다")
            inserted = vectorstore.insert_children(rows)
            if inserted != len(rows):
                raise RuntimeError("새 자식 청크 적재 건수가 일치하지 않는다")
            rebuild_bm25()
            path.unlink()
        except Exception:
            logger.exception("문서 교체 실패 → 기존 데이터 복구: %s", source_doc)
            try:
                _restore_snapshot(snapshot)
            except Exception:
                logger.exception("문서 복구 실패. 저널을 보존하고 검색·추가 적재를 차단한다")
            raise
        return deleted_children, deleted_parents


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
    """문서 갱신: 추출·청킹·임베딩 준비 → 복구 저널 → 저장소 교체.

    **같은 (project_id, source_doc)** 이 이미 적재돼 있으면 갱신이다. 파일명이
    같아도 프로젝트가 다르면 **별개 문서**라 서로 덮어쓰지 않는다.
    준비 실패는 기존 데이터를 변경하지 않는다. 교체 실패는 저널로 복구하며,
    복구까지 실패하면 검색과 추가 변경을 차단한다.

    project_id는 ""(전사 공용)가 기본이다.

    반환: {"source_doc", "project_id", "deleted_children", "deleted_parents",
          "chunks_indexed"}. 추출 실패 시 ValueError.
    """
    source_doc = path.name
    department = validate_department(owning_department)
    text, sections = load_document(path)
    if not text.strip():
        raise ValueError(f"{source_doc}: 텍스트를 추출하지 못했다 (스캔본 PDF?)")

    with _LOCK:
        ensure_store_ready()
        state = {
            "text": text,
            "source_doc": source_doc,
            "domain": domain,
            "owning_department": department,
            "visibility": visibility,
            "project_id": project_id,
            "sections": sections,
        }
        state.update(chunk(state))
        rows = prepare_rows(state)
        deleted_children, deleted_parents = _replace_document(state, rows)
    return {
        "source_doc": source_doc,
        "project_id": project_id,
        "deleted_children": deleted_children,
        "deleted_parents": deleted_parents,
        "chunks_indexed": len(rows),
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
    STORE_LOCK.acquire()
    try:
        ensure_store_ready()
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
        STORE_LOCK.release()
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
    STORE_LOCK.acquire()
    try:
        ensure_store_ready()
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
        STORE_LOCK.release()
        _LOCK.release()

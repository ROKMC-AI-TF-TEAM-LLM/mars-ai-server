"""indexer_graph: chunk → embed_and_upsert → END (architecture.md §5).

노드는 상태 dict를 받아 변경분 dict만 반환한다.
"""

from __future__ import annotations

import time
import uuid

import requests
from langgraph.graph import END, START, StateGraph

from ax_rag.indexer_graph.chunking import chunk_parent_child
from ax_rag.indexer_graph.state import IndexState
from ax_rag.shared import bm25_store, parent_store, vectorstore
from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 임베딩 서버 1회 호출당 텍스트 수 (서버 내부 배치와 별개의 요청 분할 단위).
# CPU 임베딩 환경에서 64배치가 60초 timeout을 초과하는 것을 실측 — 호출당
# 처리 시간이 timeout 안에 들도록 16으로 유지한다
_EMBED_REQUEST_BATCH = 16

# BM25 재빌드 시 corpus 메타데이터로 보존할 필드 (ACL 후처리 필터에 필요한 필드 포함)
_BM25_META_FIELDS = [
    "chunk_id",
    "parent_id",
    "source_doc",
    "domain",
    "owning_department",
    "visibility",
]


def chunk(state: IndexState) -> dict:
    """문서를 부모-자식 청크로 분할한다. 섹션이 있으면 섹션 단위 우선 (혼입 방지)."""
    sections = state.get("sections") or [{"title": None, "text": state["text"]}]
    all_parents: list[dict] = []
    all_children: list[dict] = []
    for section in sections:
        parents, children = chunk_parent_child(
            section["text"], state["source_doc"], section_title=section.get("title")
        )
        offset = len(all_children)
        for child in children:
            child["chunk_index"] += offset
        all_parents.extend(parents)
        all_children.extend(children)
    logger.info(
        "청킹 완료: %s → 부모 %d개, 자식 %d개",
        state["source_doc"],
        len(all_parents),
        len(all_children),
    )
    return {"chunks": all_children, "parents": all_parents}


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """임베딩 서버 호출 (localhost, timeout 필수)."""
    config = get_config()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_REQUEST_BATCH):
        batch = texts[start : start + _EMBED_REQUEST_BATCH]
        response = requests.post(
            config.EMBEDDING_SERVER_URL,
            json={"texts": batch},
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        embeddings.extend(response.json()["embeddings"])
    return embeddings


def _rebuild_bm25() -> None:
    """전체 자식 청크로 BM25 인덱스를 재빌드한다 (부분 갱신 불가).

    방금 insert된 청크가 조회에 보장되도록 먼저 flush한다.
    """
    vectorstore.flush()
    rows = vectorstore.fetch_all_children(output_fields=["text", *_BM25_META_FIELDS])
    texts = [row["text"] for row in rows]
    metadatas = [{field: row[field] for field in _BM25_META_FIELDS} for row in rows]
    bm25_store.build_bm25_index(texts, metadatas)


def embed_and_upsert(state: IndexState) -> dict:
    """임베딩 서버 호출 → 부모/자식 Milvus insert → BM25 인덱스 재빌드."""
    chunks = state.get("chunks") or []
    if not chunks:
        logger.warning("적재할 청크가 없다: %s", state["source_doc"])
        return {"chunks_indexed": 0}

    embeddings = _embed_texts([c["text"] for c in chunks])

    now = int(time.time())
    rows = [
        {
            "chunk_id": uuid.uuid4().hex,
            "embedding": embedding,
            "text": c["text"],
            "parent_id": c["parent_id"],
            "source_doc": state["source_doc"],
            "chunk_index": c["chunk_index"],
            "domain": state["domain"],
            "owning_department": state["owning_department"],
            "visibility": state["visibility"],
            "doc_classification": "NORMAL",  # 예약 필드: 현재 항상 NORMAL (interfaces.md §2)
            "created_at": now,
        }
        for c, embedding in zip(chunks, embeddings, strict=True)
    ]

    # 부모를 먼저 넣어 자식 parent_id 참조가 조회 불가능해지는 구간을 없앤다
    parent_store.insert_parents(state.get("parents") or [])
    inserted = vectorstore.insert_children(rows)
    _rebuild_bm25()
    logger.info("적재 완료: %s → 자식 %d건", state["source_doc"], inserted)
    return {"chunks_indexed": inserted}


def _build_graph() -> StateGraph:
    builder = StateGraph(IndexState)
    builder.add_node("chunk", chunk)
    builder.add_node("embed_and_upsert", embed_and_upsert)
    builder.add_edge(START, "chunk")
    builder.add_edge("chunk", "embed_and_upsert")
    builder.add_edge("embed_and_upsert", END)
    return builder


graph = _build_graph().compile()

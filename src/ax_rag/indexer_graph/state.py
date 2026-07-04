"""indexer_graph 상태 정의 (interfaces.md §3)."""

from typing import TypedDict


class IndexState(TypedDict):
    """문서 적재 그래프 상태.

    text~visibility는 호출자가 넣는 입력, 나머지는 노드가 채우는 파생 값이다.
    """

    text: str
    source_doc: str
    domain: str
    owning_department: str
    visibility: str
    sections: list[dict] | None  # [{"title": str|None, "text": str}, ...]
    chunks: list[dict] | None  # 자식 청크 [{"text", "chunk_index", "parent_id", "section_title"}]
    # 스펙 보충 필드: chunk 노드가 만든 부모 청크를 embed_and_upsert로 전달한다
    # (부모-자식 청킹 결과 중 부모 저장분, interfaces.md §4 chunk_parent_child 참조)
    parents: list[dict] | None  # [{"parent_id", "parent_text", "source_doc"}, ...]
    chunks_indexed: int | None

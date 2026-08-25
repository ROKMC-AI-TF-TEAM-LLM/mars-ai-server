"""dense_retrieve의 PK 추출 — Milvus 서버와 Lite의 반환 구조가 다르다.

실측 (WSL AlmaLinux 9, milvus-lite 2.4.11):
    서버 (개발 Docker) : hit["chunk_id"]                  — 실제 필드명 키
    Lite (운영 L40)    : hit["id"], entity["chunk_id"]    — PK가 "id"로 온다

한쪽만 읽으면 다른 환경에서 KeyError가 나 dense 검색이 통째로 죽는다.
"""

from __future__ import annotations

from ax_rag.query_graph.nodes.dense_retrieve import _primary_key


def test_Milvus_서버_형태_hit에_chunk_id가_있다() -> None:
    hit = {"chunk_id": "abc123", "distance": 0.9}
    assert _primary_key(hit, {"text": "본문"}) == "abc123"


def test_Milvus_Lite_형태_PK가_id로_온다() -> None:
    """★ 운영 환경 형태. 이걸 못 읽으면 KeyError로 검색이 죽는다."""
    hit = {"id": "abc123", "distance": 0.9, "entity": {"text": "본문"}}
    assert _primary_key(hit, {"text": "본문"}) == "abc123"


def test_entity_안의_chunk_id도_읽는다() -> None:
    """Lite는 output_fields로 요청하면 entity에도 담아 준다."""
    hit = {"id": "fallback", "distance": 0.9}
    assert _primary_key(hit, {"chunk_id": "abc123", "text": "본문"}) == "abc123"


def test_어디에도_없으면_빈_문자열() -> None:
    """예외 대신 빈 값 — 한 건 때문에 검색 전체가 죽지 않게 한다."""
    assert _primary_key({"distance": 0.9}, {"text": "본문"}) == ""

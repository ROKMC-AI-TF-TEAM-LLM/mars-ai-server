"""부모 청크 컬렉션 document_parents (interfaces.md §2).

리랭크 확정된 자식 청크를 생성 컨텍스트용 부모 청크로 치환할 때 조회한다.
벡터 검색은 하지 않지만, Milvus는 벡터 필드 없는 컬렉션을 허용하지 않으므로
검색에 쓰지 않는 형식상의 2차원 더미 벡터 필드를 둔다.
"""

from __future__ import annotations

from pymilvus import DataType

from ax_rag.shared.vectorstore import get_client

# 컬렉션 이름은 스펙 고정 (interfaces.md §2)
PARENT_COLLECTION = "document_parents"

# Milvus 제약(벡터 필드 필수) 회피용 더미 벡터. 검색에 사용하지 않는다
_DUMMY_DIM = 2
_DUMMY_VECTOR = [0.0, 0.0]


def get_parent_collection(drop_existing: bool = False) -> str:
    """document_parents 컬렉션을 생성한다 (이미 있으면 재사용). 컬렉션 이름 반환."""
    client = get_client()

    if drop_existing and client.has_collection(PARENT_COLLECTION):
        client.drop_collection(PARENT_COLLECTION)
    if client.has_collection(PARENT_COLLECTION):
        return PARENT_COLLECTION

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("parent_id", DataType.VARCHAR, is_primary=True, max_length=64)
    # 800~1,200토큰 분량, 한국어 여유 8000자 (interfaces.md §2)
    schema.add_field("parent_text", DataType.VARCHAR, max_length=8000)
    schema.add_field("source_doc", DataType.VARCHAR, max_length=512)
    schema.add_field("dummy_vector", DataType.FLOAT_VECTOR, dim=_DUMMY_DIM)

    index_params = client.prepare_index_params()
    # 컬렉션 로드에 벡터 인덱스가 필요해 최소 비용의 FLAT을 사용한다
    index_params.add_index(field_name="dummy_vector", index_type="FLAT", metric_type="L2")
    client.create_collection(PARENT_COLLECTION, schema=schema, index_params=index_params)
    return PARENT_COLLECTION


def insert_parents(rows: list[dict]) -> int:
    """부모 청크 rows를 insert하고 삽입 건수를 반환한다. 더미 벡터는 여기서 채운다."""
    if not rows:
        return 0
    client = get_client()
    filled = [{**row, "dummy_vector": _DUMMY_VECTOR} for row in rows]
    result = client.insert(get_parent_collection(), filled)
    return int(result["insert_count"])


def get_parent(parent_id: str) -> str:
    """parent_text 반환. 없으면 빈 문자열."""
    client = get_client()
    rows = client.query(
        get_parent_collection(),
        filter=f'parent_id == "{parent_id}"',
        output_fields=["parent_text"],
    )
    return rows[0]["parent_text"] if rows else ""


def delete_by_source_doc(source_doc: str) -> int:
    """특정 문서의 부모 청크를 전부 삭제한다 (문서 갱신용). 삭제 건수 반환."""
    client = get_client()
    result = client.delete(get_parent_collection(), filter=f'source_doc == "{source_doc}"')
    return int(result["delete_count"]) if isinstance(result, dict) else len(result)

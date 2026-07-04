"""Milvus Lite 자식 청크 컬렉션 company_docs (interfaces.md §2).

Milvus Lite는 임베디드 라이브러리라 포트가 없고, MilvusClient(로컬 파일)로
접속한다. 단일 uvicorn 워커 전제 (파일 락 충돌 방지, CLAUDE.md).

MilvusClient 기반이므로 create_collection/get_collection은 ORM Collection
객체 대신 컬렉션 이름(str)을 반환한다. 조작은 get_client()를 통해 한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pymilvus import DataType, MilvusClient

from ax_rag.shared.config import get_config

# BGE-M3 dense 차원 (interfaces.md §2)
EMBED_DIM = 1024

# Milvus query 1회 상한. 사내 문서 규모(수만 청크 미만) 전제이며,
# 초과가 확인되면 query_iterator 도입으로 전환한다
_QUERY_LIMIT = 16384


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    """Milvus 클라이언트 싱글턴.

    운영(L40)은 Milvus Lite 로컬 파일 경로를 쓴다. 개발 노트북(Windows)은
    Milvus Lite 미지원이라 localhost의 Milvus standalone URI
    (http://localhost:19530)도 허용한다. 에어갭 규칙상 URI는 localhost만 가능.
    """
    config = get_config()
    uri = config.MILVUS_LITE_PATH
    if uri.startswith("http"):
        host = urlparse(uri).hostname
        if host not in ("localhost", "127.0.0.1"):
            raise ValueError(f"Milvus URI에 localhost가 아닌 호스트는 허용되지 않는다: {uri}")
    else:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri)


def create_collection(drop_existing: bool = False) -> str:
    """company_docs 컬렉션을 생성한다 (이미 있으면 재사용). 컬렉션 이름 반환."""
    config = get_config()
    client = get_client()
    name = config.MILVUS_COLLECTION

    if drop_existing and client.has_collection(name):
        client.drop_collection(name)
    if client.has_collection(name):
        return name

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=4000)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source_doc", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("domain", DataType.VARCHAR, max_length=32)
    schema.add_field("owning_department", DataType.VARCHAR, max_length=32)
    schema.add_field("visibility", DataType.VARCHAR, max_length=16)
    # 예약 필드: 현재는 항상 "NORMAL". 향후 문서 등급-사용자 신원등급 매칭용. 삭제 금지
    schema.add_field("doc_classification", DataType.VARCHAR, max_length=16)
    schema.add_field("created_at", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # Strong 정합성: 적재 직후 BM25 재빌드용 전체 조회가 방금 insert를 봐야 한다
    client.create_collection(
        name, schema=schema, index_params=index_params, consistency_level="Strong"
    )
    return name


def get_collection() -> str:
    """존재가 보장된 company_docs 컬렉션 이름을 반환한다."""
    return create_collection(drop_existing=False)


def insert_children(rows: list[dict]) -> int:
    """자식 청크 rows를 insert하고 삽입 건수를 반환한다."""
    if not rows:
        return 0
    client = get_client()
    result = client.insert(get_collection(), rows)
    return int(result["insert_count"])


def flush() -> None:
    """insert된 데이터를 세그먼트로 확정한다 (적재 직후 전체 조회 정합성 보장)."""
    get_client().flush(get_collection())


def fetch_all_children(output_fields: list[str]) -> list[dict]:
    """모든 자식 청크를 조회한다 (BM25 전체 재빌드용)."""
    client = get_client()
    return client.query(
        get_collection(),
        filter='chunk_id != ""',
        output_fields=output_fields,
        limit=_QUERY_LIMIT,
    )


def delete_by_source_doc(source_doc: str) -> int:
    """특정 문서의 자식 청크를 전부 삭제한다 (문서 갱신용). 삭제 건수 반환."""
    client = get_client()
    result = client.delete(get_collection(), filter=f'source_doc == "{source_doc}"')
    return int(result["delete_count"]) if isinstance(result, dict) else len(result)

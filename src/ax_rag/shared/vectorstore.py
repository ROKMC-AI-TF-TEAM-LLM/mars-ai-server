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
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# BGE-M3 dense 차원 (interfaces.md §2)
EMBED_DIM = 1024

# Milvus query 1회 상한 (iterator 미지원 환경의 폴백 경로에서만 사용)
_QUERY_LIMIT = 16384

# 전체 순회 시 배치 크기
_ITERATOR_BATCH = 2000


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
    # 프로젝트 격리: ""=전사 공용, 그 외=해당 프로젝트 전용.
    # 일반 채팅은 ""만 검색하고, 프로젝트 채팅은 "" + 자기 프로젝트를 검색한다.
    # 프로젝트 문서의 접근 통제는 이 필드가 담당하므로 visibility는 ALL로 둔다
    schema.add_field("project_id", DataType.VARCHAR, max_length=64)
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
    """모든 자식 청크를 조회한다 (BM25 전체 재빌드, 문서 인벤토리용).

    query 1회 상한(16,384행)을 넘는 대규모 코퍼스를 위해 query_iterator로
    전체를 순회한다. iterator 미지원 환경(구버전/Lite 제약)에서는 단일
    query로 폴백한다 — 이 경우 16,384행까지만 조회됨을 경고한다.
    """
    client = get_client()
    name = get_collection()
    try:
        iterator = client.query_iterator(
            collection_name=name,
            filter='chunk_id != ""',
            output_fields=output_fields,
            batch_size=_ITERATOR_BATCH,
        )
    except Exception:
        logger.warning(
            "query_iterator 미지원 → 단일 query 폴백 (최대 %d행까지만 조회됨)", _QUERY_LIMIT
        )
        return client.query(
            name, filter='chunk_id != ""', output_fields=output_fields, limit=_QUERY_LIMIT
        )

    rows: list[dict] = []
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            return rows
        rows.extend(batch)


def list_documents() -> list[dict]:
    """적재 문서 인벤토리: source_doc별 도메인/공개범위/부서/청크 수/적재시각 집계.

    관리·디버깅용 (GET /documents). 무한 스크롤 페이지네이션이 안정적이도록
    문서명 오름차순 정렬로 반환한다.
    """
    rows = fetch_all_children(
        ["source_doc", "domain", "visibility", "owning_department", "project_id", "created_at"]
    )
    # 그룹 키가 **(project_id, source_doc) 복합키**다. 이름만으로 묶으면 부대마다
    # 올린 동명 문서가 한 줄로 합쳐져 청크 수가 합산되고 소속이 뒤섞인다
    documents: dict[tuple[str, str], dict] = {}
    for row in rows:
        entry = documents.setdefault(
            (row.get("project_id") or "", row["source_doc"]),
            {
                "source_doc": row["source_doc"],
                "domain": row["domain"],
                "visibility": row["visibility"],
                "owning_department": row["owning_department"],
                "project_id": row.get("project_id") or "",
                "chunk_count": 0,
                "applied_at": 0,  # unix timestamp, 청크 중 최신 적재 시각
            },
        )
        entry["chunk_count"] += 1
        entry["applied_at"] = max(entry["applied_at"], int(row.get("created_at") or 0))
    # 전사 문서를 먼저, 그다음 프로젝트별로 묶어 이름순
    return sorted(documents.values(), key=lambda d: (d["project_id"], d["source_doc"]))


def delete_by_filter(collection_name: str, expr: str) -> int:
    """필터 식에 걸리는 행을 삭제하고 건수를 반환한다 (부모 컬렉션도 함께 사용).

    pymilvus 버전에 따라 delete가 dict(delete_count) 또는 삭제된 PK 목록을
    반환해서, 두 형태를 모두 건수로 정규화한다.
    """
    result = get_client().delete(collection_name, filter=expr)
    return int(result["delete_count"]) if isinstance(result, dict) else len(result)


def _document_filter(source_doc: str, project_id: str) -> str:
    """문서 1건을 가리키는 필터. **식별자는 (project_id, source_doc) 복합키다.**

    프로젝트가 생기면서 파일명만으로는 문서가 유일하지 않다 — 부대마다 자기
    "휴가규정.md"를 올릴 수 있다. project_id 조건을 빠뜨리면 동명의 남의 프로젝트
    문서까지 지운다.
    """
    return f'source_doc == "{source_doc}" and project_id == "{project_id}"'


def fetch_parent_ids(source_doc: str, project_id: str = "") -> list[str]:
    """문서 1건에 속한 자식들의 parent_id 목록 (중복 제거).

    부모 컬렉션에는 project_id가 없으므로(parent_id로만 조회된다) 부모를 이름으로
    지우면 동명의 다른 프로젝트 부모까지 사라진다. 실제 참조 관계인 parent_id로
    지우기 위해 자식 삭제 **전에** 모아 둔다.
    """
    rows = get_client().query(
        get_collection(),
        filter=_document_filter(source_doc, project_id),
        output_fields=["parent_id"],
        limit=_QUERY_LIMIT,
    )
    return sorted({str(row.get("parent_id") or "") for row in rows} - {""})


def delete_by_source_doc(source_doc: str, project_id: str = "") -> int:
    """문서 1건의 자식 청크를 삭제한다 (문서 갱신·삭제용). 삭제 건수 반환.

    project_id 기본값 ""는 전사 공용 문서를 뜻한다 — 생략하면 프로젝트 문서는
    건드리지 않는다.
    """
    return delete_by_filter(get_collection(), _document_filter(source_doc, project_id))


def delete_by_project(project_id: str) -> int:
    """프로젝트에 속한 자식 청크를 전부 삭제한다. 삭제 건수 반환.

    ⚠️ 빈 project_id는 **전사 공용 문서 전체**를 지우게 되므로 거부한다.
    """
    if not project_id:
        raise ValueError("project_id가 비어 있다 (전사 문서 전체 삭제 방지)")
    return delete_by_filter(get_collection(), f'project_id == "{project_id}"')

"""자식 청크 컬렉션 company_docs (interfaces.md §2).

접속 모드는 MILVUS_LITE_PATH 값의 형태로 자동 결정된다 (get_client 참조).
운영·개발 모두 Milvus Lite(임베디드)를 쓴다 — 3.x가 순수 파이썬이라
Windows에서도 동작한다. Milvus 서버 URI도 계속 허용하지만(디버깅·비교용),
**동작이 미묘하게 다르므로 검증은 Lite에서 한다** (docs/milvus_lite_3x.md).

Lite는 포트가 없는 임베디드 라이브러리이므로 단일 uvicorn 워커를 전제한다
(파일 락 충돌 방지, CLAUDE.md).

MilvusClient 기반이므로 create_collection/get_collection은 ORM Collection
객체 대신 컬렉션 이름(str)을 반환한다. 조작은 get_client()를 통해 한다.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlparse

from pymilvus import DataType, MilvusClient

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger, silence_milvus_lite_noise

logger = get_logger(__name__)

# BGE-M3 dense 차원 (interfaces.md §2)
EMBED_DIM = 1024

# Milvus query 1회 상한 (iterator 미지원 환경의 폴백 경로에서만 사용)
_QUERY_LIMIT = 16384

# 전체 순회 시 배치 크기
_ITERATOR_BATCH = 2000


def is_server_uri(uri: str) -> bool:
    """접속 대상이 Milvus 서버(URI)인지 Lite 파일 경로인지 판별한다.

    같은 설정값(MILVUS_LITE_PATH)이 두 모드를 겸한다 — 운영 L40은 Lite 파일,
    개발 Windows는 Docker standalone URI다.
    """
    return uri.startswith(("http://", "https://", "tcp://"))


def _check_localhost_only(uri: str) -> None:
    """에어갭 규칙: Milvus 서버 URI는 localhost만 허용한다 (CLAUDE.md)."""
    host = urlparse(uri).hostname
    if host not in ("localhost", "127.0.0.1"):
        raise ValueError(f"Milvus URI에 localhost가 아닌 호스트는 허용되지 않는다: {uri}")


def _check_lite_available(uri: str) -> None:
    """milvus-lite가 설치돼 있는지 확인한다.

    확인 없이 MilvusClient에 넘기면 `ModuleNotFoundError: No module named
    'milvus_lite'` 라는, 원인도 해법도 알려주지 않는 오류가 난다.
    에어갭에서는 검색으로 해결할 수 없으므로 여기서 해법까지 알려 준다.
    """
    if find_spec("milvus_lite") is not None:
        return
    raise RuntimeError(
        f"milvus-lite가 설치돼 있지 않은데 파일 경로가 지정됐다: MILVUS_LITE_PATH={uri}\n"
        f"  현재 플랫폼: {sys.platform}\n"
        "  설치: pip install milvus-lite==3.2.1 faiss-cpu==1.15.0\n"
        "        (에어갭이면 반입한 wheel로: pip install --no-index --find-links <경로> ...)\n"
        "  3.x는 순수 파이썬이라 Windows·Linux 모두 동작한다 — docs/milvus_lite_3x.md 참조."
    )


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    """Milvus 클라이언트 싱글턴. 설정값 형태로 접속 모드를 자동 판별한다.

    | 값의 형태                  | 모드            | 쓰는 환경        |
    |---------------------------|-----------------|-----------------|
    | ./data/milvus_ax.db       | Lite (임베디드) | 운영 L40, WSL   |
    | http://localhost:19530    | 서버 (Docker)   | 개발 Windows    |

    두 모드의 동작이 미묘하게 다르므로(인덱스 타입 제약, search 결과의 PK 위치)
    어느 쪽으로 붙었는지 기동 로그에 남긴다 — docs/troubleshooting.md ⑩⑪.
    """
    uri = get_config().MILVUS_LITE_PATH
    if is_server_uri(uri):
        _check_localhost_only(uri)
        logger.info("Milvus 서버 모드로 접속한다: %s", uri)
    else:
        _check_lite_available(uri)
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        # Lite를 쓸 때만 나오는 gRPC 미구현 소음을 여기서 막는다.
        # setup_logging()을 부르지 않는 진입점(스크립트 등)도 있어 접속 지점에서 건다
        silence_milvus_lite_noise()
        logger.info("Milvus Lite(임베디드) 모드로 접속한다: %s", uri)
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
    # ⚠️ HNSW를 쓰면 안 된다. Milvus Lite(운영 L40)는 로컬 모드에서
    # FLAT·IVF_FLAT·AUTOINDEX만 지원하고 HNSW는 create_collection 단계에서 거부한다
    # ("invalid index type: HNSW, local mode only support ..."). 개발이 Docker
    # standalone(풀 서버)을 쓰는 탓에 이 실패가 운영에서만 드러났다.
    #
    # AUTOINDEX는 양쪽 모두에서 동작한다 — Lite는 로컬 인덱스로, 서버 모드는
    # Milvus가 코퍼스 규모에 맞는 인덱스를 알아서 고른다
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    # Strong 정합성: 적재 직후 BM25 재빌드용 전체 조회가 방금 insert를 봐야 한다
    client.create_collection(
        name, schema=schema, index_params=index_params, consistency_level="Strong"
    )
    return name


def ensure_loaded(name: str) -> str:
    """컬렉션이 로드돼 있음을 보장한다 (released 상태면 load). 이름을 그대로 반환.

    Milvus는 released 상태의 컬렉션에 search/query를 거부한다
    (`Collection X is in state 'released'; call load() before search/get/query`).
    로드는 컬렉션 생성 직후에만 유지되고, **새 프로세스가 기존 컬렉션을 열면
    released다.**

    Milvus Lite 2.4.11은 이걸 자동으로 해 줘서 그동안 호출이 없어도 문제가
    없었다. 하지만 Milvus 서버(개발 Docker)는 재시작 후 released가 되고,
    milvus-lite 3.x도 서버와 같은 시맨틱을 따른다 — 어느 쪽이든 필요한 호출이다.

    로드 상태 조회는 실패해도 무방하다(구버전은 API가 없을 수 있다). 그 경우
    그냥 load를 시도하고, 이미 로드돼 있으면 무해하게 통과한다.
    """
    client = get_client()
    try:
        state = client.get_load_state(collection_name=name)
        if str(state.get("state", state)).endswith("Loaded"):
            return name
    except Exception:  # noqa: BLE001 — 상태 조회 실패는 load 시도로 대체한다
        pass
    try:
        client.load_collection(name)
    except Exception as exc:  # noqa: BLE001 — load 불가 환경에서도 검색은 시도한다
        logger.warning("컬렉션 load 실패 (%s): %s", name, exc)
    return name


def get_collection() -> str:
    """존재가 보장되고 로드된 company_docs 컬렉션 이름을 반환한다."""
    return ensure_loaded(create_collection(drop_existing=False))


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

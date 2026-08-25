"""벡터 인덱스를 AUTOINDEX로 다시 만든다 (HNSW → AUTOINDEX).

Milvus Lite(운영 L40)는 로컬 모드에서 FLAT·IVF_FLAT·AUTOINDEX만 지원하고
HNSW는 create_collection 단계에서 거부한다. 개발이 Docker standalone(풀 서버)을
쓰는 탓에 이 실패가 운영에서만 드러난다.

인덱스 타입은 컬렉션 생성 시점에 정해지므로 컬렉션을 다시 만들어야 한다.
migrate_project_id.py와 같은 방식으로 **기존 청크를 임베딩까지 읽어와 다시 넣는다**:
재임베딩·PDF 재파싱이 없고 chunk_id·parent_id가 보존되므로 부모 컬렉션과
BM25 인덱스를 손대지 않아도 된다.

사용:
    python scripts/migrate_index_type.py --dry-run
    python scripts/migrate_index_type.py

⚠️ 컬렉션을 drop 했다가 다시 만든다. 중간에 실패하면 백업 파일로 복구한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ax_rag.shared.logging_setup import get_logger, setup_logging
from ax_rag.shared.vectorstore import (
    create_collection,
    fetch_all_children,
    get_client,
    get_collection,
    insert_children,
)

logger = get_logger(__name__)

# 새 컬렉션으로 옮길 필드 전체 (스키마 순서와 무관, 이름으로 매칭된다)
_CARRY_FIELDS = [
    "chunk_id",
    "embedding",
    "text",
    "parent_id",
    "source_doc",
    "chunk_index",
    "domain",
    "owning_department",
    "visibility",
    "project_id",
    "doc_classification",
    "created_at",
]

_INSERT_BATCH = 500
_TARGET_INDEX = "AUTOINDEX"


def _current_index_type() -> str:
    """현재 embedding 필드의 인덱스 타입. 없으면 빈 문자열."""
    try:
        return str(get_client().describe_index(get_collection(), "embedding").get("index_type", ""))
    except Exception:
        return ""


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="벡터 인덱스를 AUTOINDEX로 재생성")
    parser.add_argument("--dry-run", action="store_true", help="읽기만 하고 변경하지 않는다")
    parser.add_argument(
        "--backup", default="./data/migrate_index_backup.json", help="드롭 전 청크 백업 경로"
    )
    args = parser.parse_args()

    current = _current_index_type()
    logger.info("현재 인덱스 타입: %s", current or "(알 수 없음)")
    if current == _TARGET_INDEX:
        logger.info("이미 %s다 → 마이그레이션 불필요", _TARGET_INDEX)
        return 0

    logger.info("기존 청크를 읽는 중 (임베딩 포함)...")
    rows = fetch_all_children(_CARRY_FIELDS)
    documents = sorted({str(row.get("source_doc") or "") for row in rows})
    logger.info("대상: 청크 %d건 / 문서 %d건", len(rows), len(documents))

    if args.dry_run:
        logger.info("--dry-run: %s → %s 전환을 여기서 중단한다", current, _TARGET_INDEX)
        return 0

    backup_path = Path(args.backup)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(json.dumps(rows, ensure_ascii=False, default=float), encoding="utf-8")
    logger.info("백업 저장: %s (%.1fMB)", backup_path, backup_path.stat().st_size / 1024 / 1024)

    logger.info("컬렉션 재생성 (drop → create, 인덱스 %s)", _TARGET_INDEX)
    create_collection(drop_existing=True)

    migrated = 0
    for start in range(0, len(rows), _INSERT_BATCH):
        migrated += insert_children(rows[start : start + _INSERT_BATCH])
        logger.info("  삽입 %d/%d", migrated, len(rows))
    get_client().flush(get_collection())

    after = _current_index_type()
    logger.info("완료: %d건 이관, 인덱스 %s → %s", migrated, current, after)
    if migrated != len(rows) or after != _TARGET_INDEX:
        logger.error("검증 실패 (읽기 %d / 삽입 %d / 인덱스 %s)", len(rows), migrated, after)
        return 1
    logger.info("백업은 검증 후 삭제해도 된다: %s", backup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

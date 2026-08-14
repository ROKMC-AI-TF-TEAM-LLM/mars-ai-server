"""company_docs 컬렉션에 project_id 필드를 추가하는 마이그레이션.

milvus-lite 2.4.11은 기존 컬렉션에 필드를 추가할 수 없다 (Milvus 2.5+ 기능).
그래서 컬렉션을 새 스키마로 다시 만들어야 하는데, **문서를 재적재하지 않는다**:
기존 청크를 임베딩까지 그대로 읽어와 다시 넣는다.

이 방식의 이점:
- 임베딩 서버가 필요 없다 (재임베딩 0회)
- PDF 재파싱이 없어 청킹 결과가 달라질 위험이 없다
- chunk_id·parent_id가 보존되므로 부모 컬렉션(document_parents)과
  BM25 인덱스를 손대지 않아도 된다

기존 청크는 전부 project_id=""(전사 공용)가 되어 동작이 달라지지 않는다.

사용:
    python scripts/migrate_project_id.py --dry-run   # 계획만 출력
    python scripts/migrate_project_id.py             # 실제 수행

⚠️ 컬렉션을 drop 했다가 다시 만든다. 중간에 실패하면 백업 파일로 복구한다
   (--backup 경로, 기본 ./data/migrate_project_id_backup.json).
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

# 새 스키마로 옮길 필드 전체 (project_id 제외 — 이 마이그레이션이 채운다)
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
    "doc_classification",
    "created_at",
]

_INSERT_BATCH = 500


def _has_project_id() -> bool:
    """현재 컬렉션 스키마에 project_id가 이미 있는지."""
    client = get_client()
    name = get_collection()
    fields = client.describe_collection(name).get("fields", [])
    return any(field.get("name") == "project_id" for field in fields)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="company_docs에 project_id 필드 추가")
    parser.add_argument("--dry-run", action="store_true", help="읽기만 하고 변경하지 않는다")
    parser.add_argument(
        "--backup",
        default="./data/migrate_project_id_backup.json",
        help="드롭 전 청크 백업 경로",
    )
    args = parser.parse_args()

    if _has_project_id():
        logger.info("project_id 필드가 이미 있다 → 마이그레이션 불필요")
        return 0

    logger.info("기존 청크를 읽는 중 (임베딩 포함)...")
    rows = fetch_all_children(_CARRY_FIELDS)
    if not rows:
        logger.warning("청크가 하나도 없다 → 컬렉션만 새 스키마로 다시 만든다")

    documents = sorted({str(row.get("source_doc") or "") for row in rows})
    logger.info("대상: 청크 %d건 / 문서 %d건", len(rows), len(documents))
    for name in documents:
        count = sum(1 for row in rows if row.get("source_doc") == name)
        logger.info("  %5d청크  %s", count, name)

    if args.dry_run:
        logger.info("--dry-run: 여기서 중단한다 (변경 없음)")
        return 0

    # 백업: 드롭 후 실패하면 이 파일로 복구한다 (임베딩까지 들어 있어 용량이 크다)
    backup_path = Path(args.backup)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    logger.info("백업 저장: %s (%.1fMB)", backup_path, backup_path.stat().st_size / 1024 / 1024)

    logger.info("컬렉션을 새 스키마로 재생성한다 (drop → create)")
    create_collection(drop_existing=True)

    migrated = 0
    for start in range(0, len(rows), _INSERT_BATCH):
        batch = [{**row, "project_id": ""} for row in rows[start : start + _INSERT_BATCH]]
        migrated += insert_children(batch)
        logger.info("  삽입 %d/%d", migrated, len(rows))
    get_client().flush(get_collection())

    logger.info('마이그레이션 완료: %d건 (전부 project_id="" 전사 공용)', migrated)
    if migrated != len(rows):
        logger.error(
            "삽입 건수 불일치: 읽기 %d건 vs 삽입 %d건 — 백업으로 확인할 것", len(rows), migrated
        )
        return 1
    logger.info("백업은 검증 후 삭제해도 된다: %s", backup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

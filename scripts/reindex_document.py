"""문서 갱신 스크립트: 기존 청크 삭제 → 재적재 → BM25 전체 재빌드.

사용 예:
    python scripts/reindex_document.py --file ./raw_docs/휴가규정.md \
        --domain HR --department HR_TEAM --visibility ALL

동작 (architecture.md §9, 공용 로직 ingest.ingest_file):
1. 텍스트 추출 검증 (실패 시 기존 데이터를 지우지 않고 중단)
2. company_docs 자식 청크 + document_parents 부모 청크 삭제
3. indexer_graph로 재적재 (BM25 전체 재빌드는 embed_and_upsert 노드가 수행)

BM25는 부분 삭제가 불가하므로 전체 재빌드된다 (야간 배치 전제).
POST /documents API와 같은 경로(ingest_file)를 사용한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ax_rag.indexer_graph.ingest import ingest_file
from ax_rag.shared.config import DOMAINS
from ax_rag.shared.logging_setup import setup_logging


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="문서 갱신 (삭제 후 재적재)")
    parser.add_argument("--file", required=True, help="재적재할 .md/.txt/.pdf 파일")
    parser.add_argument("--domain", required=True, choices=DOMAINS, help="문서 도메인")
    parser.add_argument("--department", required=True, help="소유 부서 (ACL)")
    parser.add_argument(
        "--visibility", default="ALL", choices=["ALL", "DEPT_ONLY"], help="공개 범위"
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"파일이 없다: {path}", file=sys.stderr)
        return 1

    try:
        result = ingest_file(
            path,
            domain=args.domain,
            owning_department=args.department,
            visibility=args.visibility,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"{result['source_doc']}: 기존 자식 {result['deleted_children']}건, "
        f"부모 {result['deleted_parents']}건 삭제"
    )
    print(
        f"{result['source_doc']}: 자식 청크 {result['chunks_indexed']}건 재적재 (BM25 재빌드 포함)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

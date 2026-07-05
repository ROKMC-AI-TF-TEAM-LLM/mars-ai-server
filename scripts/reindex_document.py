"""문서 갱신 스크립트: 기존 청크 삭제 → 재적재 → BM25 전체 재빌드.

사용 예:
    python scripts/reindex_document.py --file ./raw_docs/휴가규정.md \
        --domain HR --department HR_TEAM --visibility ALL

동작 (architecture.md §9):
1. company_docs에서 해당 source_doc의 자식 청크 삭제
2. document_parents에서 부모 청크 삭제
3. indexer_graph로 재적재 (BM25 전체 재빌드는 embed_and_upsert 노드가 수행)

BM25는 부분 삭제가 불가하므로 전체 재빌드된다 (야간 배치 전제).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ax_rag.indexer_graph.graph import graph
from ax_rag.indexer_graph.loaders import load_document
from ax_rag.shared import parent_store, vectorstore
from ax_rag.shared.config import DOMAINS
from ax_rag.shared.logging_setup import setup_logging


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="문서 갱신 (삭제 후 재적재)")
    parser.add_argument("--file", required=True, help="재적재할 .md/.txt 파일")
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

    source_doc = path.name
    deleted_children = vectorstore.delete_by_source_doc(source_doc)
    deleted_parents = parent_store.delete_by_source_doc(source_doc)
    print(f"{source_doc}: 기존 자식 {deleted_children}건, 부모 {deleted_parents}건 삭제")

    text, sections = load_document(path)
    if not text.strip():
        print(f"{source_doc}: 텍스트를 추출하지 못했다 (스캔본 PDF?)", file=sys.stderr)
        return 1
    result = graph.invoke(
        {
            "text": text,
            "source_doc": source_doc,
            "domain": args.domain,
            "owning_department": args.department,
            "visibility": args.visibility,
            "sections": sections,
        }
    )
    print(f"{source_doc}: 자식 청크 {result.get('chunks_indexed', 0)}건 재적재 (BM25 재빌드 포함)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

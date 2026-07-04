"""여러 문서를 순회하며 indexer_graph를 호출하는 적재 스크립트.

사용 예:
    python scripts/bulk_ingest.py --dir ./raw_docs --domain HR \
        --department HR_TEAM --visibility ALL

- .md는 `## ` 헤딩 기준으로 섹션을 분해해 전달한다 (구조 인식 청킹)
- .txt는 통짜 텍스트로 전달한다
- HWP/PDF/DOCX 파서는 미확정 항목 (roadmap.md) — 현재는 텍스트 파일만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ax_rag.indexer_graph.graph import graph
from ax_rag.shared.config import DOMAINS


def parse_markdown_sections(text: str) -> list[dict] | None:
    """`## ` 헤딩 기준 섹션 분해. 헤딩이 하나도 없으면 None (통짜 처리)."""
    sections: list[dict] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({"title": current_title, "text": body})
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append({"title": current_title, "text": body})

    has_heading = any(section["title"] for section in sections)
    return sections if has_heading else None


def main() -> int:
    parser = argparse.ArgumentParser(description="문서 일괄 적재 (indexer_graph 호출)")
    parser.add_argument("--dir", required=True, help="적재할 .md/.txt 문서 디렉터리")
    parser.add_argument("--domain", required=True, choices=DOMAINS, help="문서 도메인")
    parser.add_argument("--department", required=True, help="소유 부서 (ACL)")
    parser.add_argument(
        "--visibility", default="ALL", choices=["ALL", "DEPT_ONLY"], help="공개 범위"
    )
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    if not doc_dir.is_dir():
        print(f"디렉터리가 없다: {doc_dir}", file=sys.stderr)
        return 1

    files = sorted([*doc_dir.glob("*.md"), *doc_dir.glob("*.txt")])
    if not files:
        print(f"적재할 .md/.txt 파일이 없다: {doc_dir}", file=sys.stderr)
        return 1

    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        sections = parse_markdown_sections(text) if path.suffix == ".md" else None
        result = graph.invoke(
            {
                "text": text,
                "source_doc": path.name,
                "domain": args.domain,
                "owning_department": args.department,
                "visibility": args.visibility,
                "sections": sections,
            }
        )
        indexed = result.get("chunks_indexed", 0)
        total += indexed
        print(f"{path.name}: 자식 청크 {indexed}건 적재")

    print(f"완료: 문서 {len(files)}개, 자식 청크 총 {total}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

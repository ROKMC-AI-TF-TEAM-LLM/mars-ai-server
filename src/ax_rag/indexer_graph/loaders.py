"""문서 파일 로더: 확장자별 텍스트/섹션 추출.

적재 스크립트(bulk_ingest, reindex_document)가 사용한다.

- .md  : 원문 + `## ` 헤딩 기준 섹션 (구조 인식 청킹)
- .txt : 원문 통짜 (섹션 없음)
- 텍스트 인코딩: UTF-8(BOM 허용) 우선, 실패 시 CP949 (Windows 메모장 대응)
- .pdf : pdfplumber로 페이지별 텍스트 추출 후 결합, 통짜 처리.
         스캔본(이미지) PDF는 텍스트가 안 나온다 (OCR 미지원)
- HWP/DOCX: 미확정 항목 (roadmap.md) — 아직 미지원
"""

from __future__ import annotations

from pathlib import Path

from ax_rag.indexer_graph.chunking import parse_markdown_sections
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 지원 확장자 (소문자)
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")


def _read_korean_text(path: Path) -> str:
    """텍스트 파일을 인코딩 자동 인식으로 읽는다.

    Windows 메모장 기본 저장(CP949)·BOM 붙은 UTF-8이 실무에서 흔해
    (실측: 적재 API UnicodeDecodeError) UTF-8만 강제하지 않는다.
    시도 순서가 중요하다 — CP949는 대부분의 바이트열을 디코드해버리므로
    UTF-8을 먼저 시도해야 UTF-8 문서가 CP949로 오독되지 않는다.
    """
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"텍스트 인코딩을 인식하지 못했다: {path.name} "
        "(지원: UTF-8/UTF-8 BOM/CP949. 파일을 UTF-8로 다시 저장해 주세요)"
    )


def _load_pdf_text(path: Path) -> str:
    """pdfplumber로 페이지별 텍스트를 추출해 빈 줄로 이어 붙인다."""
    # 지연 임포트: PDF를 안 쓰는 경로(유닛 테스트 등)에서 의존성 강제 방지
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)
    if not pages:
        logger.warning("PDF에서 텍스트를 추출하지 못했다 (스캔본?): %s", path.name)
    return "\n\n".join(pages)


def load_document(path: Path) -> tuple[str, list[dict] | None]:
    """(text, sections) 반환. sections가 None이면 통짜 처리.

    지원하지 않는 확장자면 ValueError.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 확장자: {path.name} (지원: {SUPPORTED_SUFFIXES})")

    if suffix == ".pdf":
        return _load_pdf_text(path), None

    text = _read_korean_text(path)
    if suffix == ".md":
        return text, parse_markdown_sections(text)
    return text, None  # .txt

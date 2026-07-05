"""indexer_graph/loaders.py 유닛 테스트 — 확장자별 디스패치와 PDF 추출."""

from __future__ import annotations

from pathlib import Path

import pytest

from ax_rag.indexer_graph.loaders import SUPPORTED_SUFFIXES, load_document


def _make_pdf(text: str) -> bytes:
    """테스트용 최소 단일 페이지 PDF를 만든다 (ASCII 전용, 외부 라이브러리 불필요)."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n" f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


def test_md는_섹션을_인식한다(tmp_path: Path) -> None:
    path = tmp_path / "규정.md"
    path.write_text(
        "# 제목\n## 연차\n연차는 15일이다.\n## 병가\n병가는 60일이다.\n", encoding="utf-8"
    )
    text, sections = load_document(path)
    assert "연차는 15일이다" in text
    assert sections is not None
    assert [s["title"] for s in sections if s["title"]] == ["연차", "병가"]


def test_txt는_통짜_처리된다(tmp_path: Path) -> None:
    path = tmp_path / "공지.txt"
    path.write_text("법인카드 한도는 직급별로 상이하다.", encoding="utf-8")
    text, sections = load_document(path)
    assert text == "법인카드 한도는 직급별로 상이하다."
    assert sections is None


def test_pdf에서_텍스트를_추출한다(tmp_path: Path) -> None:
    path = tmp_path / "policy.pdf"
    path.write_bytes(_make_pdf("Annual leave is 15 days per year."))
    text, sections = load_document(path)
    assert "Annual leave is 15 days" in text
    assert sections is None  # PDF는 통짜 처리


def test_빈_PDF는_빈_텍스트를_반환한다(tmp_path: Path) -> None:
    """스캔본(텍스트 레이어 없음) 상황: 예외가 아니라 빈 문자열 → 호출부가 건너뛴다."""
    path = tmp_path / "empty.pdf"
    path.write_bytes(_make_pdf(" "))
    text, _ = load_document(path)
    assert text.strip() == ""


def test_미지원_확장자는_거부된다(tmp_path: Path) -> None:
    path = tmp_path / "문서.hwp"
    path.write_text("한글 파일", encoding="utf-8")
    with pytest.raises(ValueError, match="지원하지 않는"):
        load_document(path)
    assert ".hwp" not in SUPPORTED_SUFFIXES

"""구조 인식 분할 + 부모-자식 청킹 (architecture.md §6).

- 토큰 수는 문자수/2.2 근사를 쓴다 (한국어, L40에서 실측 보정 예정)
- 각 청크 앞에 맥락 헤더 `[문서명 > 섹션명]`을 부착한다
- 부모(생성 컨텍스트)-자식(검색 대상) 이중 청킹: 자식은 parent_id로 부모를 참조한다
"""

from __future__ import annotations

import math
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ax_rag.shared.config import CHARS_PER_TOKEN

# 한국어 구조 인식 분할 우선순위 (architecture.md §6 단계 2)
_SEPARATORS = ["\n##", "\n###", "\n\n", "\n", "다.", "요.", ".", ""]

# 자식 청크 간 중첩 (검색 경계 손실 완화). 부모는 생성 컨텍스트 중복을 피해 중첩 없음
_CHILD_OVERLAP_TOKENS = 30


def _approx_token_len(text: str) -> int:
    """한국어 토큰 수 근사: 문자수 / 2.2 올림."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _make_header(source_doc: str, section_title: str | None) -> str:
    """맥락 헤더: 섹션이 있으면 `[문서명 > 섹션명]`, 없으면 `[문서명]`."""
    if section_title:
        return f"[{source_doc} > {section_title}]"
    return f"[{source_doc}]"


def _split(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """토큰 예산 기준 재귀 분할. 문장 종결(다./요.)은 앞 청크 끝에 남긴다."""
    # splitter는 overlap >= chunk_size를 거부하므로 1/4로 상한을 둔다
    overlap_tokens = min(overlap_tokens, chunk_size_tokens // 4)
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        length_function=_approx_token_len,
        keep_separator="end",
    )
    return [piece.strip() for piece in splitter.split_text(text) if piece.strip()]


def chunk_document(
    text: str,
    source_doc: str,
    section_title: str | None = None,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 60,
    prepend_context_header: bool = True,
) -> list[dict]:
    """단일 텍스트(또는 한 섹션)를 평면 청크 목록으로 분할한다."""
    header = _make_header(source_doc, section_title)
    chunks: list[dict] = []
    for index, piece in enumerate(_split(text, chunk_size_tokens, overlap_tokens)):
        chunk_text = f"{header}\n{piece}" if prepend_context_header else piece
        chunks.append({"text": chunk_text, "chunk_index": index, "section_title": section_title})
    return chunks


def chunk_document_by_sections(
    sections: list[dict],
    source_doc: str,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[dict]:
    """섹션 단위로 분할해 섹션 간 텍스트 혼입을 방지한다.

    chunk_index는 문서 전체에서 이어지는 순번이다.
    """
    all_chunks: list[dict] = []
    for section in sections:
        section_chunks = chunk_document(
            section["text"],
            source_doc,
            section_title=section.get("title"),
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        for chunk in section_chunks:
            chunk["chunk_index"] = len(all_chunks)
            all_chunks.append(chunk)
    return all_chunks


def chunk_parent_child(
    text: str,
    source_doc: str,
    section_title: str | None = None,
    parent_size_tokens: int = 1000,
    child_size_tokens: int = 175,
) -> tuple[list[dict], list[dict]]:
    """(parents, children) 반환. parents는 document_parents 컬렉션에,
    children은 company_docs 컬렉션에 upsert. children의 각 dict는
    parent_id로 자신이 속한 parent를 참조한다."""
    header = _make_header(source_doc, section_title)
    parents: list[dict] = []
    children: list[dict] = []
    for parent_text in _split(text, parent_size_tokens, overlap_tokens=0):
        parent_id = uuid.uuid4().hex
        parents.append(
            {"parent_id": parent_id, "parent_text": parent_text, "source_doc": source_doc}
        )
        for piece in _split(parent_text, child_size_tokens, _CHILD_OVERLAP_TOKENS):
            children.append(
                {
                    "text": f"{header}\n{piece}",
                    "chunk_index": len(children),
                    "parent_id": parent_id,
                    "section_title": section_title,
                }
            )
    return parents, children

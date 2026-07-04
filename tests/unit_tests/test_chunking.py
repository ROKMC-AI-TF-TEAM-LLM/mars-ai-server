"""indexer_graph/chunking.py 유닛 테스트 (한국어 픽스처, roadmap 2단계 DoD)."""

from __future__ import annotations

from ax_rag.indexer_graph.chunking import (
    chunk_document,
    chunk_document_by_sections,
    chunk_parent_child,
)
from ax_rag.shared.config import CHARS_PER_TOKEN

# 섹션 혼입 검증용 마커가 뚜렷한 두 섹션
_SECTION_A = (
    "연차휴가는 입사일 기준으로 매년 15일이 부여된다. "
    "사용하지 않은 연차는 다음 해로 최대 5일까지 이월할 수 있다. "
    "연차 사용 촉진 제도에 따라 미사용 연차는 소멸될 수 있다."
)
_SECTION_B = (
    "법인카드는 부서장 승인 후 발급된다. "
    "월 사용 한도는 직급별로 상이하며 경리팀이 관리한다. "
    "개인 용도 사용은 금지되며 위반 시 징계 대상이 된다."
)

_LONG_TEXT = (
    "제1조 목적. 이 규정은 임직원의 휴가 사용 기준을 정한다. "
    "제2조 연차휴가. 연차휴가는 매년 15일이 부여된다. "
    "미사용 연차는 다음 해로 이월할 수 있으며 최대 5일까지 허용된다. "
    "제3조 육아휴직. 육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있다. "
    "육아휴직 급여는 고용보험에서 지급되며 회사는 신청을 거부할 수 없다. "
    "제4조 경조휴가. 본인 결혼 시 5일, 자녀 결혼 시 1일이 부여된다. "
    "배우자 출산 시 10일의 휴가가 부여되며 분할 사용이 가능하다. "
    "제5조 병가. 업무 외 질병으로 인한 병가는 연 60일 한도로 한다. "
    "병가 사용 시 진단서 제출이 필요하며 3일 이내는 생략할 수 있다. "
) * 4


def test_맥락_헤더가_문서명과_섹션명으로_부착된다() -> None:
    chunks = chunk_document(_SECTION_A, "휴가규정.md", section_title="연차")
    assert chunks
    assert all(c["text"].startswith("[휴가규정.md > 연차]\n") for c in chunks)

    chunks_no_section = chunk_document(_SECTION_A, "휴가규정.md")
    assert all(c["text"].startswith("[휴가규정.md]\n") for c in chunks_no_section)


def test_헤더_비활성화_옵션() -> None:
    chunks = chunk_document(_SECTION_A, "휴가규정.md", prepend_context_header=False)
    assert all(not c["text"].startswith("[") for c in chunks)


def test_섹션_간_텍스트가_혼입되지_않는다() -> None:
    sections = [
        {"title": "연차휴가", "text": _SECTION_A},
        {"title": "법인카드", "text": _SECTION_B},
    ]
    # 작은 청크로 쪼개도 한 청크에 두 섹션 내용이 섞이면 안 된다
    chunks = chunk_document_by_sections(sections, "총무규정.md", chunk_size_tokens=40)
    assert len(chunks) >= 2
    for c in chunks:
        assert not ("연차" in c["text"] and "법인카드" in c["text"].split("]\n", 1)[1])
    # chunk_index는 문서 전체에서 이어지는 순번
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_한국어_문장_종결_separator로_분할된다() -> None:
    chunks = chunk_document(_LONG_TEXT, "휴가규정.md", chunk_size_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    # 청크 크기가 근사 예산(문자수/2.2)을 크게 넘지 않아야 한다 (헤더 여유 포함 1.3배)
    max_chars = int(100 * CHARS_PER_TOKEN * 1.3)
    for c in chunks:
        body = c["text"].split("]\n", 1)[1]
        assert len(body) <= max_chars
    # 대부분의 청크가 문장 종결로 끝나야 한다 (keep_separator="end")
    endings = sum(1 for c in chunks if c["text"].rstrip().endswith(("다.", "요.", ".")))
    assert endings >= len(chunks) - 1


def test_부모_자식_parent_id_참조_무결성() -> None:
    parents, children = chunk_parent_child(
        _LONG_TEXT, "휴가규정.md", parent_size_tokens=200, child_size_tokens=50
    )
    assert len(parents) >= 2  # 긴 문서는 부모가 여러 개
    assert len(children) > len(parents)  # 부모보다 자식이 많다

    parent_ids = {p["parent_id"] for p in parents}
    assert len(parent_ids) == len(parents)  # parent_id 중복 없음
    for child in children:
        assert child["parent_id"] in parent_ids  # 모든 자식이 실재하는 부모를 참조

    assert [c["chunk_index"] for c in children] == list(range(len(children)))


def test_자식_본문은_자기_부모_텍스트에_포함된다() -> None:
    parents, children = chunk_parent_child(
        _LONG_TEXT, "휴가규정.md", parent_size_tokens=200, child_size_tokens=50
    )
    parent_by_id = {p["parent_id"]: p["parent_text"] for p in parents}
    for child in children:
        body = child["text"].split("]\n", 1)[1]
        # 공백 정규화 후 부분 문자열 비교 (분할 시 strip 영향 제거)
        normalized_body = "".join(body.split())
        normalized_parent = "".join(parent_by_id[child["parent_id"]].split())
        assert normalized_body in normalized_parent


def test_부모는_한글_여유_8000자를_넘지_않는다() -> None:
    parents, _ = chunk_parent_child(_LONG_TEXT, "휴가규정.md")
    for p in parents:
        assert len(p["parent_text"]) <= 8000  # document_parents VARCHAR(8000) 상한

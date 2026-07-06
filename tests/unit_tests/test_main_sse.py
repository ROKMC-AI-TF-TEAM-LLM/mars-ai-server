"""main.py SSE 경계 유닛 테스트 — role 변환, 프레임 형식, 분할 경계 (roadmap 5단계 DoD)."""

from __future__ import annotations

import asyncio
import json

import main

# ---------- to_internal_history ----------


def test_미들웨어_role이_내부_role로_변환된다() -> None:
    messages = [
        {"role": "human", "content": "육아휴직에 대해 알려줘"},
        {"role": "ai", "content": "육아휴직은 최대 1년까지..."},
    ]
    assert main.to_internal_history(messages) == [
        {"role": "user", "content": "육아휴직에 대해 알려줘"},
        {"role": "assistant", "content": "육아휴직은 최대 1년까지..."},
    ]


def test_알_수_없는_role은_건너뛴다() -> None:
    messages = [
        {"role": "system", "content": "무시돼야 함"},
        {"role": "human", "content": "질문"},
        {"content": "role 자체가 없음"},
    ]
    converted = main.to_internal_history(messages)
    assert converted == [{"role": "user", "content": "질문"}]


# ---------- normalize_requested_domain ----------


def test_도메인_정규화_허용값만_필터로_인정() -> None:
    assert main.normalize_requested_domain("HR") == "HR"
    assert main.normalize_requested_domain("finance_legal") == "FINANCE_LEGAL"  # 대소문자 허용


def test_도메인_정규화_빈값_ALL_GENERAL_미지값은_전체_검색() -> None:
    assert main.normalize_requested_domain("") == ""
    assert main.normalize_requested_domain("ALL") == ""
    assert main.normalize_requested_domain("all") == ""
    assert main.normalize_requested_domain("GENERAL") == ""
    assert main.normalize_requested_domain("MARKETING") == ""  # 미지 값은 무시
    assert main.normalize_requested_domain("SMALLTALK") == ""  # 검색 도메인이 아님


# ---------- sse_event ----------


def test_sse_프레임_형식() -> None:
    frame = main.sse_event({"type": "text", "content": "안녕"})
    assert frame == 'data: {"type": "text", "content": "안녕"}\n\n'
    assert "\\u" not in frame  # ensure_ascii=False: 한글이 그대로 보인다


# ---------- split_for_stream ----------


def test_문장_경계로_분할된다() -> None:
    text = "연차는 15일입니다. 이월은 5일까지 가능해요. 자세한 내용은 규정을 보세요."
    pieces = main.split_for_stream(text)
    assert len(pieces) == 3
    assert pieces[0].rstrip().endswith("다.")
    assert pieces[1].rstrip().endswith("요.")


def test_분할_조각을_이어붙이면_원문과_같다() -> None:
    text = "첫 문장입니다. 둘째 문장이에요. 마지막은 마침표가 없다"
    assert "".join(main.split_for_stream(text)) == text


def test_경계가_없으면_80자_내외로_잘린다() -> None:
    text = "가" * 200  # 마침표 없음
    pieces = main.split_for_stream(text)
    assert all(len(p) <= 80 for p in pieces)
    assert "".join(pieces) == text


def test_빈_문자열은_빈_결과() -> None:
    assert main.split_for_stream("") == []


# ---------- stream_answer ----------


def _collect(final_answer: str, sources: list[dict]) -> list[str]:
    async def gather() -> list[str]:
        return [frame async for frame in main.stream_answer(final_answer, sources)]

    return asyncio.run(gather())


def test_이벤트_순서는_text_sources_done이다() -> None:
    frames = _collect(
        "육아휴직은 최대 1년입니다. 신청은 인사팀에 하세요.",
        [{"name": "휴가규정.pdf", "page": None}],
    )
    # 종료 신호는 [DONE] 문자열이 아니라 {"type": "done"} JSON 이벤트 (미들웨어 계약)
    assert json.loads(frames[-1].removeprefix("data: ").strip()) == {"type": "done"}

    payloads = [json.loads(f.removeprefix("data: ").strip()) for f in frames[:-1]]
    types = [p["type"] for p in payloads]
    assert types.count("sources") == 1  # sources는 정확히 1회
    assert types[-1] == "sources"  # done 직전
    assert all(t == "text" for t in types[:-1]) and len(types) >= 2

    # text 조각을 합치면 원문 복원
    text = "".join(p["content"] for p in payloads if p["type"] == "text")
    assert text == "육아휴직은 최대 1년입니다. 신청은 인사팀에 하세요."
    assert payloads[-1]["items"] == [{"name": "휴가규정.pdf", "page": None}]


# ---------- GET /documents ----------


def _fake_doc(index: int, domain: str = "HR") -> dict:
    return {
        "source_doc": f"문서_{index:03d}.pdf",
        "domain": domain,
        "visibility": "ALL",
        "owning_department": "HR_TEAM",
        "chunk_count": 10,
        "applied_at": 1_783_200_000,
    }


def test_문서_목록_무한_스크롤_페이지네이션(monkeypatch) -> None:
    fake_docs = [_fake_doc(i) for i in range(25)]  # 이름 오름차순으로 이미 정렬됨
    monkeypatch.setattr(main.vectorstore, "list_documents", lambda: list(fake_docs))

    first = main.list_documents(offset=0, limit=10)
    assert first.total == 25
    assert len(first.documents) == 10
    assert first.has_more is True
    assert first.documents[0].name == "문서_000.pdf"
    assert first.documents[0].type == "PDF"
    assert not hasattr(first.documents[0], "chunk_count")  # 응답에서 제외 (내부 집계용)

    second = main.list_documents(offset=10, limit=10)
    assert second.documents[0].name == "문서_010.pdf"  # 이어지는 페이지
    assert second.has_more is True

    last = main.list_documents(offset=20, limit=10)
    assert len(last.documents) == 5
    assert last.has_more is False  # 마지막 페이지


def test_문서_목록_도메인_필터(monkeypatch) -> None:
    fake_docs = [_fake_doc(0, "HR"), _fake_doc(1, "FINANCE_LEGAL"), _fake_doc(2, "HR")]
    monkeypatch.setattr(main.vectorstore, "list_documents", lambda: list(fake_docs))

    result = main.list_documents(domain="finance_legal")  # 대소문자 무관
    assert result.total == 1
    assert result.documents[0].domain == "FINANCE_LEGAL"
    assert result.has_more is False


def test_문서_목록_빈_저장소(monkeypatch) -> None:
    monkeypatch.setattr(main.vectorstore, "list_documents", lambda: [])
    result = main.list_documents()
    assert result.total == 0
    assert result.documents == []
    assert result.has_more is False


# ---------- _status_after_node ----------


def test_노드_완료에_따른_status_안내() -> None:
    assert main._status_after_node("route", {"domain": "HR"}) == (
        "retrieve",
        "사내 문서를 검색하는 중...",
    )
    assert main._status_after_node("route", {"domain": "SMALLTALK"}) == (
        "generate",
        "답변을 생성하는 중...",
    )
    assert main._status_after_node("fuse", {})[0] == "rerank"
    assert main._status_after_node("rerank", {})[0] == "generate"
    assert main._status_after_node("generate", {})[0] == "verify"
    assert main._status_after_node("increment_retry", {})[1] == "답변을 다시 생성하는 중..."


def test_안내가_없는_노드는_status를_만들지_않는다() -> None:
    for node in ("dense_retrieve", "bm25_retrieve", "verify", "finalize", "fallback", "smalltalk"):
        assert main._status_after_node(node, {}) is None


# ---------- _build_sources ----------


def test_sources는_중복_없이_page_null로_만든다() -> None:
    chunks = [
        {"text": "본문1", "source_doc": "휴가규정.pdf"},
        {"text": "본문2", "source_doc": "휴가규정.pdf"},
        {"text": "본문3", "source_doc": "경비규정.pdf"},
    ]
    assert main._build_sources(chunks, grounded=True) == [
        {"name": "휴가규정.pdf", "page": None},
        {"name": "경비규정.pdf", "page": None},
    ]


def test_검증_미통과_답변에는_sources를_붙이지_않는다() -> None:
    """fallback 답변에 검색 상위 문서가 출처로 노출되면 안 된다 (예: 잡담 질의)."""
    chunks = [{"text": "본문", "source_doc": "휴가규정.pdf"}]
    assert main._build_sources(chunks, grounded=False) == []

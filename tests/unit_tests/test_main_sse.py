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


def test_이벤트_순서는_text_sources_DONE이다() -> None:
    frames = _collect(
        "육아휴직은 최대 1년입니다. 신청은 인사팀에 하세요.",
        [{"name": "휴가규정.pdf", "page": None}],
    )
    assert frames[-1] == "data: [DONE]\n\n"

    payloads = [json.loads(f.removeprefix("data: ").strip()) for f in frames[:-1]]
    types = [p["type"] for p in payloads]
    assert types.count("sources") == 1  # sources는 정확히 1회
    assert types[-1] == "sources"  # [DONE] 직전
    assert all(t == "text" for t in types[:-1]) and len(types) >= 2

    # text 조각을 합치면 원문 복원
    text = "".join(p["content"] for p in payloads if p["type"] == "text")
    assert text == "육아휴직은 최대 1년입니다. 신청은 인사팀에 하세요."
    assert payloads[-1]["items"] == [{"name": "휴가규정.pdf", "page": None}]


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

"""HWP_EXPORT 도구 유닛 테스트 — HWPX 생성기, 결정적 매처, 노드, 다운로드 API."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import main
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from ax_rag.query_graph.graph import after_route
from ax_rag.query_graph.nodes.hwp_export import (
    NO_CONTENT_ANSWER,
    hwp_export,
    is_hwp_export_request,
)
from ax_rag.shared.config import get_config
from ax_rag.shared.hwpx_writer import write_hwpx


@pytest.fixture()
def export_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """EXPORT_DIR을 임시 폴더로 바꾸고 config 캐시를 격리한다."""
    monkeypatch.setenv("EXPORT_DIR", str(tmp_path))
    get_config.cache_clear()
    yield tmp_path
    get_config.cache_clear()


# ---------- hwpx_writer ----------


def test_hwpx는_필수_엔트리를_가진_유효한_zip이다(tmp_path: Path) -> None:
    path = write_hwpx("제목", "첫 문단\n둘째 문단", tmp_path / "문서.hwpx")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "mimetype"  # 컨테이너 규약: 첫 엔트리
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED  # 무압축
        assert archive.read("mimetype").decode() == "application/hwp+zip"
        for required in (
            "version.xml",
            "META-INF/container.xml",
            "Contents/content.hpf",
            "Contents/header.xml",
            "Contents/section0.xml",
            "settings.xml",
        ):
            assert required in names
            # 모든 XML이 정형(well-formed)이어야 한다
            if required.endswith((".xml", ".hpf")):
                ElementTree.fromstring(archive.read(required))


def test_hwpx_본문_텍스트가_문단으로_들어간다(tmp_path: Path) -> None:
    body = "연차는 15일이다.\n특수문자 <검증> & 이스케이프"
    path = write_hwpx("답변", body, tmp_path / "문서.hwpx")
    with zipfile.ZipFile(path) as archive:
        section = archive.read("Contents/section0.xml").decode("utf-8")
    assert "연차는 15일이다." in section
    assert "&lt;검증&gt; &amp; 이스케이프" in section  # XML 이스케이프
    root = ElementTree.fromstring(section)
    texts = [t.text for t in root.iter("{http://www.hancom.co.kr/hwpml/2011/paragraph}t")]
    assert "답변" in texts  # 제목 문단 포함


# ---------- 결정적 매처 ----------


def test_매처는_한글파일_생성_요청만_잡는다() -> None:
    assert is_hwp_export_request("이 답변 한글 파일로 저장해줘") is True
    assert is_hwp_export_request("방금 내용 hwp로 만들어줘") is True
    assert is_hwp_export_request("한글 문서로 내보내줘") is True
    # 사용법·절차 질문은 문서 검색으로
    assert is_hwp_export_request("한글 문서 작성 방법 알려줘") is False
    # 생성 동사가 없으면 미매치
    assert is_hwp_export_request("한글 파일이 뭐야?") is False
    assert is_hwp_export_request("휴가 규정 알려줘") is False


# ---------- 도구 노드 ----------


def test_이전_답변이_없으면_안내만_한다(export_dir: Path) -> None:
    result = hwp_export({"question": "한글 파일로 저장해줘", "conversation_history": []})
    assert result["final_answer"] == NO_CONTENT_ANSWER
    assert result["grounded"] is False
    assert list(export_dir.iterdir()) == []  # 파일 미생성


def test_직전_답변을_hwpx로_저장하고_다운로드_링크를_답한다(export_dir: Path) -> None:
    history = [
        {"role": "user", "content": "육아휴직 얼마나 써?"},
        {"role": "assistant", "content": "육아휴직은 최대 1년까지 사용할 수 있습니다."},
    ]
    result = hwp_export({"question": "한글 파일로 저장해줘", "conversation_history": history})

    assert result["grounded"] is False  # 문서 근거 주장 아님 (sources 미노출)
    # 다운로드 경로는 텍스트가 아니라 SSE file 이벤트로만 전달한다 (미들웨어 신호)
    assert len(result["generated_files"]) == 1
    assert result["generated_files"][0]["tool"] == "HWP_EXPORT"
    assert result["generated_files"][0]["url"].startswith("/files/")
    files = list(export_dir.glob("*.hwpx"))
    assert len(files) == 1
    assert result["generated_files"][0]["name"] == files[0].name
    with zipfile.ZipFile(files[0]) as archive:
        section = archive.read("Contents/section0.xml").decode("utf-8")
    assert "육아휴직은 최대 1년까지" in section  # 직전 답변이 본문에 담김


def test_단독_계획도_HWP_EXPORT_노드로_라우팅된다() -> None:
    assert after_route({"intents": ["HWP_EXPORT"]}) == "HWP_EXPORT"


# ---------- 후처리(검색 + 파일 생성 복합) ----------


def test_실행_큐에서_후처리_도구는_검색_뒤에_온다() -> None:
    from ax_rag.query_graph.tools import execution_queue

    assert execution_queue(["HWP_EXPORT", "DOC_SEARCH"]) == ["DOC_SEARCH", "HWP_EXPORT"]
    assert execution_queue(["HWP_EXPORT", "DOC_SEARCH", "DISCHARGE_DAYS"]) == [
        "DISCHARGE_DAYS",
        "DOC_SEARCH",
        "HWP_EXPORT",
    ]


def test_finalize는_큐의_DOC_SEARCH를_소비하고_후처리로_넘긴다() -> None:
    from langgraph.graph import END

    from ax_rag.query_graph.graph import after_finalize, finalize

    result = finalize(
        {"draft_answer": "검증된 답변", "pending_intents": ["DOC_SEARCH", "HWP_EXPORT"]}
    )
    assert result["pending_intents"] == ["HWP_EXPORT"]
    assert after_finalize(result) == "HWP_EXPORT"
    assert after_finalize({"pending_intents": []}) == END  # 후처리 없으면 종료


def test_후처리_래퍼는_확정_답변_뒤에_도구_답변을_붙인다() -> None:
    from ax_rag.query_graph import graph as graph_module

    step = graph_module._make_post_tool_step(
        "HWP_EXPORT",
        lambda state: {
            "final_answer": "다운로드: /files/문서.hwpx",
            "generated_files": [
                {"name": "문서.hwpx", "url": "/files/문서.hwpx", "tool": "HWP_EXPORT"}
            ],
        },
    )
    result = step(
        {
            "final_answer": "육아휴직은 최대 1년입니다.",
            "pending_intents": ["HWP_EXPORT"],
        }
    )
    assert result["final_answer"] == "육아휴직은 최대 1년입니다.\n\n다운로드: /files/문서.hwpx"
    assert result["pending_intents"] == []
    # 파일 정보가 래퍼를 통과해 SSE file 이벤트까지 전달된다
    assert result["generated_files"] == [
        {"name": "문서.hwpx", "url": "/files/문서.hwpx", "tool": "HWP_EXPORT"}
    ]


def test_방금_확정된_답변이_있으면_그걸_내보낸다(export_dir: Path) -> None:
    """복합 질문(검색 → 파일 생성): 대화 이력이 아니라 방금 검증된 답변을 담는다."""
    result = hwp_export(
        {
            "question": "휴가 규정 찾아서 한글 파일로 저장해줘",
            "final_answer": "연차휴가는 매년 15일이 부여됩니다.",
            "conversation_history": [{"role": "assistant", "content": "옛날 답변"}],
        }
    )
    assert result["generated_files"][0]["url"].startswith("/files/")
    files = list(export_dir.glob("*.hwpx"))
    assert len(files) == 1
    with zipfile.ZipFile(files[0]) as archive:
        section = archive.read("Contents/section0.xml").decode("utf-8")
    assert "연차휴가는 매년 15일이" in section  # 방금 답변이 담김
    assert "옛날 답변" not in section  # 이력이 아니라 확정 답변 우선


# ---------- EXPORT_DIR TTL 정리 ----------


def test_TTL_지난_산출물은_새_파일_생성_시_정리된다(export_dir: Path) -> None:
    """기회적 정리: 디렉터리가 커지는 유일한 경로(생성)에서 만료분을 지운다."""
    import os
    import time as time_module

    expired_file = export_dir / "만료된_옛파일.hwpx"
    expired_file.write_bytes(b"old")
    expired_at = time_module.time() - 25 * 3600  # TTL(24시간)보다 오래됨
    os.utime(expired_file, (expired_at, expired_at))
    fresh_file = export_dir / "최근파일.hwpx"
    fresh_file.write_bytes(b"new")

    history = [{"role": "assistant", "content": "육아휴직은 최대 1년입니다."}]
    hwp_export({"question": "한글 파일로 저장해줘", "conversation_history": history})

    names = {p.name for p in export_dir.iterdir()}
    assert "만료된_옛파일.hwpx" not in names  # 만료분 삭제
    assert "최근파일.hwpx" in names  # TTL 이내는 유지
    assert any(n.startswith("MARS_답변_") for n in names)  # 새 파일 생성


def test_TTL_0이면_정리하지_않는다(monkeypatch: pytest.MonkeyPatch, export_dir: Path) -> None:
    import os
    import time as time_module

    from ax_rag.shared.exports import cleanup_expired_exports

    monkeypatch.setenv("EXPORT_TTL_HOURS", "0")
    get_config.cache_clear()
    old_file = export_dir / "아주_옛날.hwpx"
    old_file.write_bytes(b"x")
    ancient = time_module.time() - 999 * 3600
    os.utime(old_file, (ancient, ancient))

    assert cleanup_expired_exports() == 0
    assert old_file.exists()  # 비활성 시 보존


# ---------- GET /files 다운로드 ----------


def test_다운로드는_EXPORT_DIR의_파일만_서빙한다(export_dir: Path) -> None:
    (export_dir / "문서.hwpx").write_bytes(b"dummy")
    response = main.download_file("문서.hwpx")
    assert isinstance(response, FileResponse)
    assert Path(response.path).name == "문서.hwpx"


def test_다운로드_없는_파일은_404() -> None:
    get_config.cache_clear()
    with pytest.raises(HTTPException) as exc:
        main.download_file("없는파일.hwpx")
    assert exc.value.status_code == 404


def test_다운로드_경로_탈출은_400() -> None:
    for bad in ("../secrets.txt", "..\\..\\config.py", ".hidden"):
        with pytest.raises(HTTPException) as exc:
            main.download_file(bad)
        assert exc.value.status_code == 400

"""HWP_EXPORT 도구 유닛 테스트 — HWPX 생성기, 결정적 매처, 노드, 다운로드 API."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import pytest

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

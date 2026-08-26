"""Milvus 접속 모드 판별 — 같은 설정값이 Lite 파일과 서버 URI를 겸한다.

운영 L40·WSL은 Lite(임베디드 파일), 개발 Windows는 milvus-lite 휠이 없어
Docker standalone URI를 쓴다. 환경이 안 맞으면 기동 시점에 명확히 실패해야
한다 — 그러지 않으면 `ModuleNotFoundError: No module named 'milvus_lite'`
라는, 원인도 해법도 알려주지 않는 오류만 남는다 (troubleshooting.md ⑫).
"""

from __future__ import annotations

import pytest

from ax_rag.shared.vectorstore import _check_lite_available, _check_localhost_only, is_server_uri


@pytest.mark.parametrize(
    "uri",
    ["http://localhost:19530", "https://localhost:19530", "tcp://127.0.0.1:19530"],
)
def test_URI_형태는_서버_모드로_판별한다(uri: str) -> None:
    assert is_server_uri(uri) is True


@pytest.mark.parametrize(
    "path",
    ["./data/milvus_ax.db", "/root/mars/data/milvus_ax.db", "C:/mars/data/milvus_ax.db"],
)
def test_파일_경로는_Lite_모드로_판별한다(path: str) -> None:
    """Windows 드라이브 문자(C:/)를 스킴으로 오인하면 안 된다."""
    assert is_server_uri(path) is False


def test_localhost가_아닌_Milvus_서버는_거부한다() -> None:
    """에어갭 규칙 — 외부 호스트로 나가는 경로를 만들지 않는다 (CLAUDE.md)."""
    with pytest.raises(ValueError, match="localhost가 아닌"):
        _check_localhost_only("http://10.0.0.5:19530")


@pytest.mark.parametrize("uri", ["http://localhost:19530", "http://127.0.0.1:19530"])
def test_localhost_Milvus_서버는_허용한다(uri: str) -> None:
    _check_localhost_only(uri)


def test_milvus_lite가_없으면_해법을_알려주며_실패한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ Windows에서 파일 경로를 지정한 경우. 오류 메시지가 해법을 담아야 한다."""
    monkeypatch.setattr("ax_rag.shared.vectorstore.find_spec", lambda name: None)
    with pytest.raises(RuntimeError) as exc:
        _check_lite_available("./data/milvus_ax.db")

    message = str(exc.value)
    assert "./data/milvus_ax.db" in message  # 어떤 값이 문제인지
    assert "docker compose" in message  # 무엇을 해야 하는지
    assert "MILVUS_LITE_PATH=http://localhost:19530" in message  # 어떻게 고치는지


def test_milvus_lite가_있으면_통과한다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ax_rag.shared.vectorstore.find_spec", lambda name: object())
    _check_lite_available("./data/milvus_ax.db")

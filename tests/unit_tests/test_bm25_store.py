"""shared/bm25_store.py 유닛 테스트 (한국어 픽스처, 외부 서비스 불필요)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ax_rag.shared import bm25_store
from ax_rag.shared.config import get_config

_TEXTS = [
    "[휴가규정.md > 연차] 연차휴가는 매년 15일이 부여되며 최대 5일까지 이월할 수 있다.",
    "[휴가규정.md > 육아휴직] 육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있다.",
    "[경비규정.md > 법인카드] 법인카드 월 사용 한도는 직급별로 상이하다.",
]
_METAS = [
    {"chunk_id": "c1", "source_doc": "휴가규정.md", "owning_department": "HR", "visibility": "ALL"},
    {"chunk_id": "c2", "source_doc": "휴가규정.md", "owning_department": "HR", "visibility": "ALL"},
    {
        "chunk_id": "c3",
        "source_doc": "경비규정.md",
        "owning_department": "FIN",
        "visibility": "DEPT_ONLY",
    },
]


@pytest.fixture()
def bm25_index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """임시 디렉터리를 BM25_INDEX_PATH로 쓰고, 설정/로드 캐시를 격리한다."""
    index_dir = tmp_path / "bm25_index"
    monkeypatch.setenv("BM25_INDEX_PATH", str(index_dir))
    get_config.cache_clear()
    bm25_store._clear_cache()
    yield index_dir
    get_config.cache_clear()
    bm25_store._clear_cache()


def test_토큰화는_내용어만_남긴다() -> None:
    tokens = bm25_store.tokenize("육아휴직은 자녀 1명당 최대 1년까지 사용할 수 있다.")
    assert "육아" in tokens or "육아휴직" in tokens
    assert "은" not in tokens  # 조사 제거
    assert "." not in tokens  # 구두점 제거


def test_빌드_후_검색하면_관련_문서가_먼저_나온다(bm25_index_dir: Path) -> None:
    bm25_store.build_bm25_index(_TEXTS, _METAS)
    assert (bm25_index_dir / "corpus.jsonl").is_file()

    results = bm25_store.bm25_search("육아휴직 기간은 얼마나 되나요?", top_k=3)
    assert results
    assert results[0]["chunk_id"] == "c2"
    # 메타데이터가 결과에 보존된다 (ACL 후처리 필터에 필요)
    assert results[0]["owning_department"] == "HR"
    assert results[0]["visibility"] == "ALL"
    assert results[0]["bm25_score"] > 0


def test_top_k가_코퍼스보다_커도_안전하다(bm25_index_dir: Path) -> None:
    bm25_store.build_bm25_index(_TEXTS, _METAS)
    results = bm25_store.bm25_search("법인카드 한도", top_k=20)
    assert 1 <= len(results) <= len(_TEXTS)
    assert results[0]["chunk_id"] == "c3"


def test_외부_프로세스가_인덱스를_갱신하면_자동_재로드된다(bm25_index_dir: Path) -> None:
    """보안 회귀 테스트: reindex 스크립트(별도 프로세스)가 ACL 메타데이터를
    갱신했는데 서버가 낡은 캐시로 검색하면 DEPT_ONLY가 노출될 수 있다.
    빌드 버전(uuid)이 바뀌면 캐시를 자동 재로드해야 한다
    (mtime 비교는 연속 재빌드 시 타임스탬프 충돌로 플레이크가 났었다)."""
    bm25_store.build_bm25_index(_TEXTS, _METAS)
    bm25_store.bm25_search("육아휴직", top_k=3)  # 캐시 적재
    stale = bm25_store._cached

    # 별도 프로세스의 재빌드 시뮬레이션: visibility 변경 후 캐시를 낡은 값으로 되돌린다
    new_metas = [{**m, "visibility": "DEPT_ONLY"} for m in _METAS]
    bm25_store.build_bm25_index(_TEXTS, new_metas)
    bm25_store._cached = stale

    results = bm25_store.bm25_search("육아휴직 기간", top_k=3)
    assert results
    assert all(r["visibility"] == "DEPT_ONLY" for r in results)  # 새 메타데이터로 재로드됨


def test_인덱스가_없으면_None과_빈_결과(bm25_index_dir: Path) -> None:
    assert bm25_store.load_bm25_index() is None
    assert bm25_store.bm25_search("육아휴직") == []


def test_길이_불일치는_거부된다(bm25_index_dir: Path) -> None:
    with pytest.raises(ValueError, match="길이"):
        bm25_store.build_bm25_index(_TEXTS, _METAS[:1])

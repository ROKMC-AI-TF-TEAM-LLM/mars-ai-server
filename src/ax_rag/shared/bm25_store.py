"""Kiwi 토큰화 + bm25s 인덱스 (하이브리드 검색의 키워드 축).

- 인덱스는 부분 갱신이 불가하므로 적재/갱신 시 전체 재빌드한다 (야간 배치 전제)
- 인덱스가 없으면 load가 None을 반환하고, 검색은 빈 리스트를 반환한다
  (retrieval_graph는 dense 단독으로 폴백)
- 주의: BM25 결과에는 Milvus ACL 필터가 적용되지 않으므로, 호출 측에서
  반드시 filter_by_acl() 후처리를 거쳐야 한다 (retrieval_graph/acl.py)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import bm25s
from kiwipiepy import Kiwi

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# 검색 신호가 되는 내용어 품사 접두 (명사/대명사/수사/동사/형용사/어근/외국어/숫자/한자)
_CONTENT_TAG_PREFIXES = ("NN", "NP", "NR", "VV", "VA", "XR", "SL", "SN", "SH")

# 인덱스 디렉터리 내 파일 배치
_INDEX_SUBDIR = "index"  # bm25s retriever.save 산출물
_CORPUS_FILE = "corpus.jsonl"  # 원문 + 메타데이터 (행 순서 = bm25s 문서 인덱스)


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """Kiwi 형태소 분석기 싱글턴 (로드 비용 절감)."""
    return Kiwi()


def tokenize(text: str) -> list[str]:
    """내용어 형태소만 추출해 소문자로 정규화한다.

    Kiwi는 문맥에 따라 복합명사 분절이 달라질 수 있다 (예: 코퍼스에서는
    "육아휴직" 한 토큰, 질의에서는 "육아"+"휴직"). 이 불일치로 매칭이
    전멸하는 것을 막기 위해, 3글자 이상 명사는 원형 토큰에 더해 문자
    bigram 조각도 함께 색인한다.
    """
    tokens: list[str] = []
    for t in _get_kiwi().tokenize(text, split_complex=True):
        if not t.tag.startswith(_CONTENT_TAG_PREFIXES):
            continue
        form = t.form.lower()
        tokens.append(form)
        if t.tag.startswith("NN") and len(form) >= 3:
            tokens.extend(form[i : i + 2] for i in range(len(form) - 1))
    return tokens


def build_bm25_index(texts: list[str], metadatas: list[dict]) -> None:
    """Kiwi 토큰화 → bm25s 인덱스 생성 → 디스크 저장(BM25_INDEX_PATH)."""
    if len(texts) != len(metadatas):
        raise ValueError(f"texts({len(texts)})와 metadatas({len(metadatas)}) 길이가 다르다")

    index_dir = Path(get_config().BM25_INDEX_PATH)
    index_dir.mkdir(parents=True, exist_ok=True)

    corpus_tokens = [tokenize(text) for text in texts]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(index_dir / _INDEX_SUBDIR))

    with open(index_dir / _CORPUS_FILE, "w", encoding="utf-8") as f:
        for text, metadata in zip(texts, metadatas, strict=True):
            f.write(json.dumps({"text": text, "meta": metadata}, ensure_ascii=False) + "\n")

    _clear_cache()
    logger.info("BM25 인덱스 재빌드 완료: 문서 %d건, 경로 %s", len(texts), index_dir)


def load_bm25_index() -> bm25s.BM25 | None:
    """디스크에서 인덱스 로드. 없으면 None (dense 단독 폴백)."""
    index_dir = Path(get_config().BM25_INDEX_PATH) / _INDEX_SUBDIR
    if not index_dir.is_dir():
        return None
    return bm25s.BM25.load(str(index_dir))


# (corpus mtime, retriever, corpus) — mtime이 바뀌면 자동 재로드한다.
# 주의: lru_cache처럼 영구 캐시하면 별도 프로세스(reindex 스크립트)가 갱신한
# ACL 메타데이터를 서버가 계속 낡은 값으로 들고 있어 DEPT_ONLY가 노출될 수 있다
_cached: tuple[float, bm25s.BM25, list[dict]] | None = None


def _clear_cache() -> None:
    """로드 캐시 무효화 (재빌드 직후, 테스트 격리용)."""
    global _cached
    _cached = None


def _load_cached() -> tuple[bm25s.BM25, list[dict]] | None:
    """(retriever, corpus 항목 목록) 로드. corpus 파일 mtime 변경 시 재로드.

    인덱스가 없으면 None (dense 단독 폴백).
    """
    global _cached
    corpus_path = Path(get_config().BM25_INDEX_PATH) / _CORPUS_FILE
    if not corpus_path.is_file():
        logger.warning("BM25 인덱스가 없다. dense 단독 폴백으로 동작한다")
        return None

    mtime = corpus_path.stat().st_mtime
    if _cached is not None and _cached[0] == mtime:
        return _cached[1], _cached[2]

    retriever = load_bm25_index()
    if retriever is None:
        return None
    with open(corpus_path, encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    _cached = (mtime, retriever, corpus)
    logger.info("BM25 인덱스 로드: 문서 %d건 (mtime=%s)", len(corpus), mtime)
    return retriever, corpus


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """Kiwi 토큰화 → bm25s 검색 → 메타데이터 포함 결과 반환.

    반환 dict는 메타데이터 필드 + text + bm25_score. 인덱스가 없거나
    질의에서 내용어가 안 나오면 빈 리스트.
    """
    loaded = _load_cached()
    if loaded is None:
        return []
    retriever, corpus = loaded

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    k = min(top_k, len(corpus))
    indices, scores = retriever.retrieve([query_tokens], k=k, show_progress=False)
    results: list[dict] = []
    for doc_index, score in zip(indices[0], scores[0], strict=True):
        entry = corpus[int(doc_index)]
        results.append({**entry["meta"], "text": entry["text"], "bm25_score": float(score)})
    return results

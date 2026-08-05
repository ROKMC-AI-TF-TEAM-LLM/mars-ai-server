"""검색 정확도 단독 평가 — 생성·검증을 빼고 리랭크 결과만 본다.

compare_answers.py의 source_hit_rate는 서버 SSE의 sources를 보는데, 검증에
실패하면(grounded=False) sources가 비워져 나온다(_build_sources). 그래서 그
지표는 pass_rate의 종속값이 되어 **실패 원인이 검색인지 생성·검증인지 구분하지
못한다** (실측: 30문항 전부에서 두 값이 일치).

이 스크립트는 route → dense → bm25 → fuse → rerank 까지만 in-process로 돌려
기대 근거 문서가 실제로 검색됐는지 직접 확인한다. generate·verify를 호출하지
않으므로 빠르고(문항당 10~30초), 검색 문제와 검증 문제를 분리해 준다.

사용 예:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --trials 2 --questions eval_sets/qualitative_questions.jsonl

지표:
- hit@n     : 기대 문서가 리랭크 확정 근거(top_n)에 들어온 비율
- hit@fuse  : 리랭크 이전 융합 후보에는 있었는지 (있는데 hit@n이 0이면 리랭크·임계값 문제)
- empty     : 근거 0건으로 끝난 비율 (임계값에서 전멸)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ax_rag.query_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.query_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.query_graph.nodes.fuse import fuse
from ax_rag.query_graph.nodes.rerank import rerank
from ax_rag.query_graph.nodes.router import route
from ax_rag.shared.config import DOMAINS, get_config
from ax_rag.shared.logging_setup import setup_logging


def _run_once(row: dict) -> dict:
    """문항 1건의 검색 결과 (리랭크 확정본 + 융합 후보)."""
    domain = row.get("domain", "")
    state: dict = {
        "question": row["question"],
        "user_department": row.get("user_department", ""),
        "requested_domain": domain if domain in DOMAINS and domain != "GENERAL" else "",
        "conversation_history": [],
    }
    state.update(route(state))
    for node in (dense_retrieve, bm25_retrieve, fuse):
        state.update(node(state))
    fused_docs = [c.get("source_doc", "") for c in state.get("retrieved_candidates") or []]
    state.update(rerank(state))
    chunks = state.get("retrieved_chunks") or []
    return {
        "queries": state.get("search_queries") or [],
        "final_docs": [c.get("source_doc", "") for c in chunks],
        "fused_docs": fused_docs,
        "top_score": round(max((c.get("rerank_score", 0.0) for c in chunks), default=0.0), 3),
    }


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="검색 정확도 단독 평가")
    parser.add_argument("--questions", default="eval_sets/qualitative_questions.jsonl")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output-dir", default="eval_sets/results")
    parser.add_argument("--label", default="retrieval")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 기대 근거가 지정된 문항만 평가한다 (안전장치 문항은 검색 대상이 아니다)
    rows = [r for r in rows if r.get("expected_source")]
    threshold = get_config().RERANK_SCORE_THRESHOLD
    print(f"검색 평가: {len(rows)}문항 × {args.trials}회 (임계값 {threshold})\n")

    results: dict[str, dict] = {}
    print(f"{'문항':<11}{'hit@n':>7}{'hit@fuse':>10}{'근거0':>7}{'최고점':>8}  검색 쿼리")
    print("-" * 78)
    for row in rows:
        expected = row["expected_source"]
        hits, fuse_hits, empties, scores = [], [], [], []
        queries: list[str] = []
        for _ in range(args.trials):
            out = _run_once(row)
            queries = out["queries"]
            hits.append(any(expected in d for d in out["final_docs"]))
            fuse_hits.append(any(expected in d for d in out["fused_docs"]))
            empties.append(not out["final_docs"])
            scores.append(out["top_score"])
        n = args.trials
        results[row["id"]] = {
            "expected": expected,
            "hit_at_n": round(sum(hits) / n, 2),
            "hit_at_fuse": round(sum(fuse_hits) / n, 2),
            "empty_rate": round(sum(empties) / n, 2),
            "avg_top_score": round(sum(scores) / n, 3),
        }
        r = results[row["id"]]
        flag = ""
        if r["hit_at_n"] == 0 and r["hit_at_fuse"] > 0:
            flag = "  << 융합엔 있는데 리랭크에서 탈락"
        elif r["hit_at_n"] == 0:
            flag = "  << 검색 자체가 못 찾음"
        print(
            f"{row['id']:<11}{r['hit_at_n']:>6.0%}{r['hit_at_fuse']:>10.0%}"
            f"{r['empty_rate']:>7.0%}{r['avg_top_score']:>8.3f}  {queries}{flag}"
        )

    total = len(results)
    summary = {
        "hit_at_n": round(sum(r["hit_at_n"] for r in results.values()) / total, 3),
        "hit_at_fuse": round(sum(r["hit_at_fuse"] for r in results.values()) / total, 3),
        "empty_rate": round(sum(r["empty_rate"] for r in results.values()) / total, 3),
        "threshold": threshold,
    }
    print("\n=== 집계 ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"retrieval_{args.label}.json"
    path.write_text(
        json.dumps({"summary": summary, "questions": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"결과 저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

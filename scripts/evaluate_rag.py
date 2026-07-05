"""RAGAS 기반 RAG 성능 평가 (requirements-eval.txt 별도 설치 필요).

evaluator LLM은 로컬 vLLM 엔드포인트(get_llm), 임베딩은 로컬 임베딩 서버를
사용한다 (에어갭: 외부 API 없음).

사용 예:
    # 하이브리드 + 리랭커 (기본)
    python scripts/evaluate_rag.py --eval-set eval_sets/hr_sample.jsonl

    # dense 단독 vs 하이브리드, 리랭커 ON/OFF, RRF k 스윕 비교
    python scripts/evaluate_rag.py --mode dense --reranker off
    python scripts/evaluate_rag.py --rrf-k 20 --label k20
    python scripts/evaluate_rag.py --rrf-k 100 --label k100

평가셋 형식(JSONL): {"question": str, "ground_truth": str,
                     "user_department": str(선택), "domain": str(선택)}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ax_rag.retrieval_graph.fusion import rrf_fuse
from ax_rag.retrieval_graph.nodes.bm25_retrieve import bm25_retrieve
from ax_rag.retrieval_graph.nodes.dense_retrieve import dense_retrieve
from ax_rag.retrieval_graph.nodes.generate import generate
from ax_rag.retrieval_graph.nodes.rerank import rerank as rerank_node
from ax_rag.shared import parent_store
from ax_rag.shared.config import get_config
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import setup_logging

TOP_N_WITHOUT_RERANKER = 5  # 리랭커 OFF일 때 RRF 순서 그대로 상위 5개 사용


def _substitute_parents(candidates: list[dict]) -> list[dict]:
    """리랭커 OFF 경로: 후보를 부모 청크로 치환한다 (중복 부모 제거)."""
    chunks: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        if len(chunks) >= TOP_N_WITHOUT_RERANKER:
            break
        parent_id = candidate.get("parent_id") or ""
        if parent_id in seen:
            continue
        parent_text = parent_store.get_parent(parent_id) if parent_id else ""
        chunks.append(
            {"text": parent_text or candidate["text"], "source_doc": candidate["source_doc"]}
        )
        if parent_id:
            seen.add(parent_id)
    return chunks


def run_pipeline(
    question: str,
    user_department: str,
    domain: str,
    mode: str,
    use_reranker: bool,
    rrf_k: int,
) -> tuple[str, list[str]]:
    """평가 변형(모드/리랭커/k)을 적용해 (답변, 컨텍스트 목록)을 만든다.

    라우터/verify는 평가 대상 축이 아니므로 생략하고 검색-생성만 실행한다.
    """
    state: dict = {
        "question": question,
        "rewritten_query": question,
        "user_department": user_department,
        # 검색 노드는 requested_domain만 본다 (라우터 분류는 검색 범위 미제한)
        "requested_domain": domain if domain in ("HR", "TECH", "FINANCE_LEGAL") else "",
        "domain": domain,
    }
    dense = dense_retrieve(state)["dense_candidates"]
    bm25 = bm25_retrieve(state)["bm25_candidates"] if mode == "hybrid" else []
    fused = rrf_fuse(dense, bm25, k=rrf_k, top_n=20)

    if use_reranker:
        chunks = rerank_node({**state, "retrieved_candidates": fused})["retrieved_chunks"]
    else:
        chunks = _substitute_parents(fused)

    draft = generate({**state, "retrieved_chunks": chunks})["draft_answer"]
    return draft, [chunk["text"] for chunk in chunks]


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="RAGAS 기반 RAG 평가")
    parser.add_argument("--eval-set", default="eval_sets/hr_sample.jsonl")
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--reranker", choices=["on", "off"], default="on")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k (스윕: 20/40/60/100)")
    parser.add_argument("--label", default=None, help="결과 파일 라벨")
    parser.add_argument("--output-dir", default="eval_sets/results")
    args = parser.parse_args()

    # ragas는 requirements-eval.txt 전용이므로 지연 임포트한다
    try:
        from langchain_core.embeddings import Embeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print(
            "ragas가 없다. 먼저 설치할 것: pip install -r requirements-eval.txt",
            file=sys.stderr,
        )
        return 1

    import requests

    class RemoteEmbeddings(Embeddings):
        """로컬 임베딩 서버(8001)를 쓰는 LangChain Embeddings 어댑터."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            config = get_config()
            response = requests.post(
                config.EMBEDDING_SERVER_URL,
                json={"texts": texts},
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["embeddings"]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    eval_path = Path(args.eval_set)
    if not eval_path.is_file():
        print(f"평가셋이 없다: {eval_path}", file=sys.stderr)
        return 1
    rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    use_reranker = args.reranker == "on"
    print(f"평가 시작: {len(rows)}문항, mode={args.mode}, reranker={args.reranker}, k={args.rrf_k}")

    samples = []
    for i, row in enumerate(rows, start=1):
        answer, contexts = run_pipeline(
            question=row["question"],
            user_department=row.get("user_department", ""),
            domain=row.get("domain", "GENERAL"),
            mode=args.mode,
            use_reranker=use_reranker,
            rrf_k=args.rrf_k,
        )
        samples.append(
            {
                "user_input": row["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": row["ground_truth"],
            }
        )
        print(f"  [{i}/{len(rows)}] {row['question'][:30]}... 완료")

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(get_llm()),
        embeddings=LangchainEmbeddingsWrapper(RemoteEmbeddings()),
    )
    print("\n=== RAGAS 결과 ===")
    print(result)

    label = args.label or f"{args.mode}_rerank-{args.reranker}_k{args.rrf_k}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"mode": args.mode, "reranker": args.reranker, "rrf_k": args.rrf_k},
                "scores": {k: float(v) for k, v in result._repr_dict.items()},
                "n_samples": len(samples),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"결과 저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

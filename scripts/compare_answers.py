"""정성 평가용 답변 수집 스크립트 (변경 전/후 비교).

RAGAS 자동 평가(evaluate_rag.py)는 route·verify를 건너뛰므로 라우팅·검증이
걸린 변경은 측정하지 못한다. 이 스크립트는 **실제 서버의 POST /query**를
그대로 호출해 전체 파이프라인을 통과한 답변을 남긴다.

★ 반복 측정이 핵심이다. 같은 문항이 실행마다 통과/실패로 뒤집히는 것을
반복 관찰했다 (LLM 판정 변동). 1회 측정으로는 5/13과 7/13의 차이를 판별할 수
없으므로, --trials로 여러 번 돌려 **통과율**로 비교한다.

사용 예:
    # 변경 전 기준선 (문항당 3회, 3병렬)
    python scripts/compare_answers.py --label before --trials 3 --concurrency 3

    # 변경 후
    python scripts/compare_answers.py --label after --trials 3 --concurrency 3

★ 병렬 실행은 운영(vLLM)에서만 쓴다. 개발 llama.cpp는 동시 요청이 오면
컨텍스트가 부족해져 generate에서 500(Context size has been exceeded)이 나고,
그 실패가 "빈 답변"으로 수집돼 측정을 오염시킨다 (실측: 30문항 중 13문항 오류,
순차로 돌리면 정상). 프롬프트 자체는 약 5,000토큰으로 12,288 예산 안에 있다.

    # 두 결과 비교
    python scripts/compare_answers.py --diff before after

전제: MARS 서버(:9000)와 로컬 서비스 4종이 기동되어 있어야 한다.
에어갭: 호출 대상은 config.MARS_SERVER_URL (기본 localhost:9000)뿐이다.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import setup_logging

# 질문 1건당 상한. CPU 리랭커 환경에서 400초를 실측해 넉넉히 잡는다
_REQUEST_TIMEOUT_SECONDS = 600

# fallback 답변을 식별하는 문구 (prompts.FALLBACK_* 와 짝)
_FALLBACK_MARKS = ("충분한 근거를 찾지 못해", "검색 범위가")


def _collect(payload: dict, base_url: str) -> dict:
    """POST /query의 SSE 스트림을 소비해 답변·출처·단계를 모은다."""
    started = time.monotonic()
    result: dict = {"answer": "", "sources": [], "files": [], "stages": [], "error": None}
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/query",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            stream=True,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            kind = event.get("type")
            if kind == "text":
                result["answer"] += event.get("content", "")
            elif kind == "sources":
                result["sources"] = [item.get("name") for item in event.get("items", [])]
            elif kind == "file":
                result["files"].append(event.get("name"))
            elif kind == "status":
                result["stages"].append(event.get("stage"))
            elif kind == "error":
                result["error"] = event.get("message")
    except requests.RequestException as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    result["elapsed"] = time.monotonic() - started
    result["fallback"] = any(mark in result["answer"] for mark in _FALLBACK_MARKS)
    return result


def _score(row: dict, runs: list[dict]) -> dict:
    """문항 1개의 반복 실행 결과를 지표로 집계한다.

    - answer_rate: 답변을 낸 비율 (expect=answer 문항의 성공률)
    - guard_rate:  fallback이 정답인 문항이 실제로 막힌 비율
    - source_hit:  기대 근거 문서가 sources에 들어온 비율
    """
    expect_answer = row.get("expect", "answer") == "answer"
    expected = row.get("expected_source", "")
    # 오류·빈 답변은 어느 쪽 기대든 실패다. fallback 문구가 없다고 통과로 세면
    # 파이프라인 예외가 "정상 답변"으로 집계된다 (실측: 빈 답변 문항이 통과율 100%)
    ok = [
        bool(r["answer"].strip()) and not r["error"] and (r["fallback"] is not expect_answer)
        for r in runs
    ]
    hits = [any(expected in (name or "") for name in r["sources"]) for r in runs if expected]
    return {
        "type": row.get("type", ""),
        "expect": "answer" if expect_answer else "fallback",
        "trials": len(runs),
        "pass_rate": round(sum(ok) / len(runs), 2),
        "source_hit_rate": round(sum(hits) / len(hits), 2) if hits else None,
        "avg_chars": round(sum(len(r["answer"]) for r in runs) / len(runs)),
        "avg_elapsed": round(sum(r["elapsed"] for r in runs) / len(runs), 1),
        "errors": sum(1 for r in runs if r["error"]),
    }


def _render(rows: list[dict], scores: dict, runs: dict, label: str, trials: int) -> str:
    """수집 결과를 비교하기 좋은 마크다운으로 만든다."""
    lines = [
        f"# 정성 평가 결과 — {label}",
        "",
        f"- 수집 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 문항 {len(rows)}개 × {trials}회",
        "",
        "## 요약",
        "",
        "| ID | 유형 | 기대 | 통과율 | 근거적중 | 평균 길이 | 평균 초 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        s = scores[row["id"]]
        hit = "-" if s["source_hit_rate"] is None else f"{s['source_hit_rate']:.0%}"
        lines.append(
            f"| {row['id']} | {s['type']} | {s['expect']} | {s['pass_rate']:.0%} | "
            f"{hit} | {s['avg_chars']} | {s['avg_elapsed']:.0f} |"
        )

    lines += ["", "---", ""]
    for row in rows:
        s = scores[row["id"]]
        hit = "-" if s["source_hit_rate"] is None else f"{s['source_hit_rate']:.0%}"
        lines += [
            f"## {row['id']} — {s['type']}",
            "",
            f"**질문**: {row['question']}",
            "",
            f"**기대**: {row.get('note', '(없음)')}",
            "",
            f"**통과율** {s['pass_rate']:.0%} ({trials}회) · **근거적중** {hit} · "
            f"**평균** {s['avg_chars']}자 / {s['avg_elapsed']:.0f}초",
            "",
        ]
        for index, run in enumerate(runs[row["id"]], start=1):
            if run["error"]:
                verdict = "오류"
            elif run["fallback"]:
                verdict = "fallback"
            else:
                verdict = "답변"
            lines += [
                f"<details><summary>{index}회차 — {verdict} "
                f"({len(run['answer'])}자, 출처 {len(run['sources'])}건)</summary>",
                "",
                "```",
                run["answer"] or "(빈 답변)",
                "```",
                f"출처: {', '.join(run['sources']) if run['sources'] else '(없음)'}",
            ]
            if run["error"]:
                lines.append(f"오류: {run['error']}")
            lines += ["", "</details>", ""]
        lines += ["---", ""]
    return "\n".join(lines)


def _aggregate(rows: list[dict], scores: dict) -> dict:
    """전체 집계 — 기대 유형별로 나눠 본다 (섞으면 원인이 안 보인다)."""
    answer_q = [r for r in rows if r.get("expect", "answer") == "answer"]
    guard_q = [r for r in rows if r.get("expect", "answer") == "fallback"]
    hits = [
        scores[r["id"]]["source_hit_rate"]
        for r in rows
        if scores[r["id"]]["source_hit_rate"] is not None
    ]
    return {
        "answer_rate": round(sum(scores[r["id"]]["pass_rate"] for r in answer_q) / len(answer_q), 3)
        if answer_q
        else None,
        "guard_rate": round(sum(scores[r["id"]]["pass_rate"] for r in guard_q) / len(guard_q), 3)
        if guard_q
        else None,
        "source_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "avg_chars": round(sum(scores[r["id"]]["avg_chars"] for r in rows) / len(rows)),
        "avg_elapsed": round(sum(scores[r["id"]]["avg_elapsed"] for r in rows) / len(rows), 1),
    }


def _load(label: str, output_dir: Path) -> dict:
    path = output_dir / f"qualitative_{label}.json"
    if not path.is_file():
        raise SystemExit(f"결과가 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _diff(before: str, after: str, output_dir: Path) -> int:
    """두 측정 결과를 지표별로 비교한다."""
    a, b = _load(before, output_dir), _load(after, output_dir)
    print(f"\n{'지표':<16}{before:>12}{after:>12}{'변화':>10}")
    print("-" * 52)
    for key, fmt in (
        ("answer_rate", "{:.0%}"),
        ("guard_rate", "{:.0%}"),
        ("source_hit_rate", "{:.0%}"),
        ("avg_chars", "{:.0f}자"),
        ("avg_elapsed", "{:.0f}초"),
    ):
        x, y = a["summary"].get(key), b["summary"].get(key)
        if x is None or y is None:
            continue
        delta = y - x
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"{key:<16}{fmt.format(x):>12}{fmt.format(y):>12}{arrow} {abs(delta):>7.2f}")

    print(f"\n{'문항':<11}{'통과율 ' + before:>14}{'통과율 ' + after:>14}  비고")
    print("-" * 56)
    for qid, sa in a["questions"].items():
        sb = b["questions"].get(qid)
        if sb is None:
            continue
        d = sb["pass_rate"] - sa["pass_rate"]
        note = "[개선]" if d > 0.1 else ("[후퇴]" if d < -0.1 else "")
        print(f"{qid:<11}{sa['pass_rate']:>13.0%}{sb['pass_rate']:>14.0%}  {note}")
    return 0


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="정성 평가용 답변 수집 (변경 전/후 비교)")
    parser.add_argument("--questions", default="eval_sets/qualitative_questions.jsonl")
    parser.add_argument("--label", help="결과 파일 라벨 (예: before, after_step1)")
    parser.add_argument("--trials", type=int, default=3, help="문항당 반복 횟수 (변동성 흡수)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="동시 실행 수. 기본 1 — 개발 llama.cpp는 동시 요청 시 컨텍스트가 "
        "부족해져 500(Context size has been exceeded)이 난다. 운영 vLLM"
        "(--max-model-len 12288 --max-num-seqs 16)에서만 올릴 것",
    )
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="두 결과 비교")
    parser.add_argument("--output-dir", default="eval_sets/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.diff:
        return _diff(args.diff[0], args.diff[1], output_dir)
    if not args.label:
        print("--label 또는 --diff 중 하나가 필요하다")
        return 1

    questions_path = Path(args.questions)
    if not questions_path.is_file():
        print(f"질문셋이 없다: {questions_path}")
        return 1
    rows = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    base_url = get_config().MARS_SERVER_URL
    total = len(rows) * args.trials
    print(
        f"수집 시작: {len(rows)}문항 × {args.trials}회 = {total}건 "
        f"(병렬 {args.concurrency}) → {base_url}"
    )

    jobs = [(row, trial) for row in rows for trial in range(args.trials)]
    done = 0

    def run(job: tuple[dict, int]) -> tuple[str, dict]:
        row, _ = job
        payload = {
            "question": row["question"],
            "user_department": row.get("user_department", ""),
            "domain": row.get("domain", ""),
        }
        return row["id"], _collect(payload, base_url)

    runs: dict[str, list[dict]] = {row["id"]: [] for row in rows}
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for qid, result in pool.map(run, jobs):
            runs[qid].append(result)
            done += 1
            mark = "실패" if result["error"] else ("fallback" if result["fallback"] else "답변")
            print(f"  [{done}/{total}] {qid}: {mark} ({result['elapsed']:.0f}초)")

    scores = {row["id"]: _score(row, runs[row["id"]]) for row in rows}
    summary = _aggregate(rows, scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"qualitative_{args.label}.md").write_text(
        _render(rows, scores, runs, args.label, args.trials), encoding="utf-8"
    )
    (output_dir / f"qualitative_{args.label}.json").write_text(
        json.dumps(
            {"summary": summary, "questions": scores, "trials": args.trials},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n=== 집계 ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"결과 저장: {output_dir}/qualitative_{args.label}.{{md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

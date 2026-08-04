"""정성 평가용 답변 수집 스크립트 (변경 전/후 비교).

RAGAS 자동 평가(evaluate_rag.py)는 route·verify를 건너뛰므로 라우팅·검증이
걸린 변경은 측정하지 못한다. 이 스크립트는 **실제 서버의 POST /query**를
그대로 호출해 전체 파이프라인을 통과한 답변을 마크다운으로 남긴다.

사용 예:
    # 변경 전 기준선
    python scripts/compare_answers.py --label before

    # 변경 후
    python scripts/compare_answers.py --label after_step1

    # 두 결과를 나란히 열어 육안 비교
    # eval_sets/results/qualitative_before.md  vs  ..._after_step1.md

전제: MARS 서버(:9000)와 로컬 서비스 4종이 기동되어 있어야 한다.
에어갭: 호출 대상은 config.MARS_SERVER_URL (기본 localhost:9000)뿐이다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from ax_rag.shared.config import get_config
from ax_rag.shared.logging_setup import setup_logging

# 질문 1건당 상한. CPU 리랭커 환경(실측 19초)에서도 여유가 있도록 넉넉히 잡는다
_REQUEST_TIMEOUT_SECONDS = 300


def _collect(payload: dict, base_url: str) -> dict:
    """POST /query의 SSE 스트림을 소비해 답변·출처·단계를 모은다.

    반환: {"answer", "sources", "files", "stages", "elapsed", "error"}
    """
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
    return result


def _render(rows: list[dict], results: list[dict], label: str) -> str:
    """수집 결과를 비교하기 좋은 마크다운으로 만든다."""
    lines = [
        f"# 정성 평가 결과 — {label}",
        "",
        f"- 수집 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 문항 수: {len(rows)}",
        "",
        "## 요약",
        "",
        "| ID | 유형 | 답변 길이 | 출처 수 | 소요(초) | 비고 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row, res in zip(rows, results, strict=True):
        note = res["error"] or ("출처 없음" if not res["sources"] else "")
        lines.append(
            f"| {row['id']} | {row.get('type', '')} | {len(res['answer'])} | "
            f"{len(res['sources'])} | {res['elapsed']:.1f} | {note} |"
        )

    lines += ["", "---", ""]
    for row, res in zip(rows, results, strict=True):
        lines += [
            f"## {row['id']} — {row.get('type', '')}",
            "",
            f"**질문**: {row['question']}",
            "",
            f"**기대**: {row.get('note', '(없음)')}",
            "",
            f"**도메인 한정**: `{row.get('domain', '') or '(전체)'}`  ·  "
            f"**소요**: {res['elapsed']:.1f}초  ·  "
            f"**단계**: {' → '.join(res['stages']) or '(없음)'}",
            "",
            "**답변**:",
            "",
            "```",
            res["answer"] or "(빈 답변)",
            "```",
            "",
            f"**출처**: {', '.join(res['sources']) if res['sources'] else '(없음)'}",
        ]
        if res["files"]:
            lines.append(f"**생성 파일**: {', '.join(res['files'])}")
        if res["error"]:
            lines.append(f"**오류**: {res['error']}")
        lines += ["", "---", ""]
    return "\n".join(lines)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="정성 평가용 답변 수집 (변경 전/후 비교)")
    parser.add_argument("--questions", default="eval_sets/qualitative_questions.jsonl")
    parser.add_argument("--label", required=True, help="결과 파일 라벨 (예: before, after_step1)")
    parser.add_argument("--output-dir", default="eval_sets/results")
    args = parser.parse_args()

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
    print(f"수집 시작: {len(rows)}문항 → {base_url}")

    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        payload = {
            "question": row["question"],
            "user_department": row.get("user_department", ""),
            "domain": row.get("domain", ""),
        }
        res = _collect(payload, base_url)
        results.append(res)
        mark = "실패" if res["error"] else f"{len(res['answer'])}자/출처 {len(res['sources'])}건"
        print(f"  [{index}/{len(rows)}] {row['id']}: {mark} ({res['elapsed']:.1f}초)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qualitative_{args.label}.md"
    output_path.write_text(_render(rows, results, args.label), encoding="utf-8")

    # 기계 판독용 지표. 마크다운을 되파싱하면 답변 속 "## 제목"에 걸려
    # 블록이 잘못 잘린다 (실측) — 비교는 이 JSON을 쓴다
    metrics_path = output_dir / f"qualitative_{args.label}.json"
    metrics_path.write_text(
        json.dumps(
            {
                row["id"]: {
                    "type": row.get("type", ""),
                    "answer_chars": len(res["answer"]),
                    "sources": res["sources"],
                    "source_count": len(res["sources"]),
                    "retried": res["stages"].count("verify") > 1,
                    "stages": res["stages"],
                    "elapsed": round(res["elapsed"], 1),
                    "error": res["error"],
                }
                for row, res in zip(rows, results, strict=True)
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"결과 저장: {output_path} / {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

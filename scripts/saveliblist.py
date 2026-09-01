"""반입 번들의 wheel 목록을 CSV로 뽑는다 (반입 승인 서류용).

세 폴더(shared/window/linux)를 한 번에 처리해 scripts/*_wheel_list.csv 를 갱신한다.
sdist(.tar.gz)도 함께 잡는다 — FlagEmbedding·kiwipiepy_model이 sdist라 빠지면
목록과 실물이 어긋난다.

사용:
    python scripts/saveliblist.py                  # 기본 경로
    python scripts/saveliblist.py <번들_최상위_경로>
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from packaging.utils import parse_wheel_filename

DEFAULT_BUNDLE = Path("C:/Users/User/Desktop/3차 라이브러리 정리")
OUT_DIR = Path(__file__).parent

# 폴더명 → 출력 CSV 접두사
FOLDERS = {
    "marsai_shared": "shared",
    "marsai_window": "window",
    "marsai_linux": "linux",
}

FIELDS = ["파일명", "패키지", "버전", "Python", "ABI", "플랫폼", "크기(KB)"]

# sdist는 wheel 파서가 못 읽으므로 파일명에서 직접 뽑는다
_SDIST_RE = re.compile(r"^(?P<name>.+?)-(?P<ver>\d[^-]*)\.(tar\.gz|zip)$", re.I)


def _row(path: Path) -> dict | None:
    size_kb = round(path.stat().st_size / 1024, 2)
    if path.suffix == ".whl":
        try:
            name, version, _build, tags = parse_wheel_filename(path.name)
            tag = next(iter(tags))
            return {
                "파일명": path.name,
                "패키지": str(name),
                "버전": str(version),
                "Python": tag.interpreter,
                "ABI": tag.abi,
                "플랫폼": tag.platform,
                "크기(KB)": size_kb,
            }
        except Exception as exc:  # noqa: BLE001 — 파싱 실패 파일은 건너뛰고 알린다
            print(f"  분석 실패: {path.name} / {exc}")
            return None

    match = _SDIST_RE.match(path.name)
    if not match:
        return None
    return {
        "파일명": path.name,
        "패키지": match.group("name"),
        "버전": match.group("ver"),
        "Python": "sdist",
        "ABI": "-",
        "플랫폼": "-",
        "크기(KB)": size_kb,
    }


def main() -> int:
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"번들 경로가 없다: {bundle}")
        return 1

    total = 0
    for folder, prefix in FOLDERS.items():
        src = bundle / folder
        if not src.exists():
            print(f"[건너뜀] {folder} — 폴더 없음")
            continue

        rows = [r for r in (_row(p) for p in sorted(src.iterdir()) if p.is_file()) if r]
        out = OUT_DIR / f"{prefix}_wheel_list.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        size_mb = sum(r["크기(KB)"] for r in rows) / 1024
        print(f"[{prefix:<7}] {len(rows):>4}개 / {size_mb:>8.1f} MB  → {out.name}")
        total += len(rows)

    print(f"\n총 {total}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import csv
from packaging.utils import parse_wheel_filename

# 현재 폴더
folder = Path("C:/Users/User/Desktop/3차 라이브러리 정리/marsai_shared")

# 결과 CSV 파일
output_file = "wheel_list.csv"

rows = []

for wheel in folder.glob("*.whl"):
    try:
        # Wheel 파일명 분석
        name, version, build, tags = parse_wheel_filename(wheel.name)

        # 태그 중 첫 번째 사용
        tag = next(iter(tags))

        rows.append({
            "파일명": wheel.name,
            "패키지": str(name),
            "버전": str(version),
            "Python": tag.interpreter,
            "ABI": tag.abi,
            "플랫폼": tag.platform,
            "크기(KB)": round(wheel.stat().st_size / 1024, 2)
        })

    except Exception as e:
        print(f"분석 실패: {wheel.name} / {e}")


# CSV 저장
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "파일명",
            "패키지",
            "버전",
            "Python",
            "ABI",
            "플랫폼",
            "크기(KB)"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"완료! {len(rows)}개의 wheel 파일을 {output_file}에 저장했습니다.")
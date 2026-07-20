"""순수 파이썬 HWPX(한글 2014+ 표준, OWPML/KS X 6101) 문서 생성기.

한글(한컴오피스)이 실제로 생성한 빈 문서(resources/hwpx_blank.hwpx,
hwpxlib 프로젝트의 검증 샘플)를 템플릿으로 쓰고, 본문 구역(section0.xml)의
문단만 우리 내용으로 치환한다 — header.xml의 방대한 참조 정의(35KB)를
손으로 만들면 한글이 빈 화면으로 여는 것을 실측해서, 검증된 골격을
그대로 재사용하는 방식으로 바꿨다.

외부 라이브러리 없이 표준 라이브러리(zipfile)만 사용한다 — 에어갭·
의존성 == 고정 규칙(CLAUDE.md) 준수.

지원 범위: 제목 1개 + 본문 문단들(개행 분리). 표·이미지·서식 채우기는
추후 확장 (docs/interfaces.md 참조).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# 한글이 생성한 빈 문서 템플릿 (모든 엔트리를 그대로 복사하고 section0만 교체)
_TEMPLATE_PATH = Path(__file__).parent / "resources" / "hwpx_blank.hwpx"
_SECTION_ENTRY = "Contents/section0.xml"

# 추가 문단 id 시작값 (한글은 문단 id를 무부호 정수로 쓴다 — 서로 다르기만 하면 됨)
_PARA_ID_BASE = 3000000000


def _build_section(template_section: str, title: str, body: str) -> str:
    """템플릿 section0.xml의 문단을 우리 내용으로 치환한 XML을 만든다.

    - 첫 문단(구역 속성 secPr·단 설정 ctrl 포함)은 템플릿 그대로 두고
      빈 텍스트(<hp:t/>)에 제목만 채운다
    - 본문 문단들은 템플릿의 lineseg(줄 배치 정보)를 복사한 단순 문단으로
      뒤에 이어 붙인다 (한글이 열 때 실제 배치는 다시 계산한다)
    """
    close_tag = "</hs:sec>"
    prefix = template_section[: template_section.rindex(close_tag)]

    # 첫 문단: 빈 텍스트 자리에 제목 주입 (템플릿의 빈 문단은 <hp:t/> 하나뿐)
    if "<hp:t/>" not in prefix:
        raise ValueError("템플릿 section0.xml에서 빈 텍스트(<hp:t/>)를 찾지 못했다")
    prefix = prefix.replace("<hp:t/>", f"<hp:t>{escape(title)}</hp:t>", 1)

    # 템플릿 문단의 줄 배치 정보를 그대로 복사해 쓴다
    start = template_section.index("<hp:linesegarray>")
    end = template_section.index("</hp:linesegarray>") + len("</hp:linesegarray>")
    lineseg = template_section[start:end]

    paragraphs: list[str] = []
    lines = ["", *body.splitlines()] if body else [""]  # 제목-본문 사이 빈 줄
    for index, line in enumerate(lines):
        paragraphs.append(
            f'<hp:p id="{_PARA_ID_BASE + index}" paraPrIDRef="3" styleIDRef="0" '
            f'pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="0"><hp:t>{escape(line)}</hp:t></hp:run>'
            f"{lineseg}</hp:p>"
        )
    return prefix + "".join(paragraphs) + close_tag


def write_hwpx(title: str, body: str, out_path: Path) -> Path:
    """제목 + 본문(개행으로 문단 분리)을 HWPX 파일로 저장한다.

    본문 텍스트는 XML 이스케이프만 하고 그대로 담는다 (내용 변형 금지).
    반환: 저장된 파일 경로.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(_TEMPLATE_PATH) as template,
        zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        for info in template.infolist():  # 순서 보존 (mimetype이 첫 엔트리)
            if info.filename == _SECTION_ENTRY:
                section = _build_section(template.read(info).decode("utf-8"), title, body)
                archive.writestr(info.filename, section)
            else:
                # 컨테이너 규약: mimetype은 무압축(STORED). 나머지는 원본 방식 유지
                compress = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
                archive.writestr(
                    zipfile.ZipInfo(info.filename),
                    template.read(info),
                    compress_type=compress,
                )
    return out_path

"""에이전트 판단 근거(thought)의 위생 처리.

thought는 **verify를 거치지 않고 사용자 화면에 그대로 표시되는 텍스트**다.
SMALLTALK·HWP_DRAFT와 같은 등급(검증 밖 LLM 자유 생성)으로 다루며,
프롬프트 지시에만 의존하지 않고 코드로 다음을 강제한다:

- 길이 상한 (긴 문장은 곧 내용 서술이다 — 행동의 이유는 짧다)
- 개행·제어문자 제거 (SSE JSON 한 줄에 실린다)
- 꺾쇠 제거 (관측 안의 delimiter가 thought로 새어 나오는 것 차단)

**금지 사항의 최종 방어선은 이 파일이 아니다.** "규정·수치를 쓰지 마라"는
프롬프트가 담당하고, 여기서는 형태만 다듬는다. 내용 통제까지 정규식으로
하려 들면 정상 문장을 깎는다 (verify 규칙 검증을 걷어낸 것과 같은 이유).
문제가 생기면 config.STREAM_THOUGHTS로 경로 자체를 끈다.
"""

from __future__ import annotations

import re

# 표시 상한. 60자 안팎을 프롬프트로 요청하고, 넘치면 여기서 자른다
MAX_THOUGHT_CHARS = 200

# 제어문자·개행 → 공백 (SSE 프레임은 JSON 한 줄이다)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
# delimiter 흉내 차단: <document>·<observation> 등이 그대로 실려 나가지 않게
_ANGLE_RE = re.compile(r"[<>]")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def sanitize_thought(raw: object) -> str:
    """LLM이 만든 thought를 사용자에게 보여도 되는 한 줄 문자열로 다듬는다.

    다듬은 결과가 비면 빈 문자열을 반환한다 — 호출부가 기본 안내 문구로
    대체하므로 스트림이 끊기지 않는다.
    """
    text = str(raw or "")
    text = _CONTROL_RE.sub(" ", text)
    text = _ANGLE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if len(text) > MAX_THOUGHT_CHARS:
        text = text[:MAX_THOUGHT_CHARS].rstrip() + "…"
    return text

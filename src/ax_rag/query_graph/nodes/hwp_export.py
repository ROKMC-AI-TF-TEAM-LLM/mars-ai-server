"""hwp_export 도구 노드: 직전 답변을 한글(HWPX) 문서 파일로 내보낸다.

- 파일 생성은 LLM 없이 결정적 코드로만 한다 (verify 밖 경로 원칙)
- 후처리 도구(POST_SEARCH_TOOLS): 복합 질문("휴가 규정 찾아서 한글로
  저장해줘")이면 검색 파이프라인 뒤에 실행되어 **방금 검증·확정된 답변**
  (state.final_answer)을 내보낸다. 단독 요청("이 답변 저장해줘")이면
  대화 이력의 마지막 assistant 답변을 쓴다. 둘 다 없으면 안내만 한다.
  검증 실패(fallback) 시에는 실행되지 않는다 (graph.after_finalize)
- 산출물은 EXPORT_DIR에 저장되고 GET /files/{파일명}으로 다운로드한다
"""

from __future__ import annotations

import re

from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tool_contract import tool_answer
from ax_rag.shared.exports import save_export_document
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

# "한글 파일/한글 문서/hwp(x)" 언급
_HWP_RE = re.compile(r"한글\s*(파일|문서)|hwpx?", re.IGNORECASE)
# 생성 요청 동사
_ACTION_RE = re.compile(r"만들|저장|내보내|변환|뽑아|출력")
# 사용법 질문("한글 파일 만드는 방법 알려줘")은 문서 검색으로 보낸다
_HOWTO_RE = re.compile(r"방법|어떻게|절차")

NO_CONTENT_ANSWER = (
    "한글 문서로 저장할 이전 답변이 없습니다. 먼저 질문해 답변을 받으신 뒤 "
    '"이 답변을 한글 파일로 저장해줘"라고 요청해 주세요.'
)

EXPORT_FAIL_ANSWER = "한글 문서 생성에 실패했습니다. 잠시 후 다시 시도해 주세요."


def is_hwp_export_request(question: str) -> bool:
    """LLM 없이 판정하는 결정적 매처 (tools.TOOL_MATCHERS 등록용)."""
    if not _HWP_RE.search(question):
        return False
    if _HOWTO_RE.search(question):
        return False  # 사용법·절차 질문은 문서 검색 경로 유지
    return bool(_ACTION_RE.search(question))


def _last_assistant_answer(history: list[dict]) -> str:
    """대화 이력에서 마지막 assistant 답변을 찾는다. 없으면 빈 문자열."""
    for message in reversed(history or []):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


def hwp_export(state: QueryState) -> dict:
    """답변을 HWPX로 저장하고 다운로드 링크를 답한다. grounded 값은 건드리지 않는다.

    내보낼 내용 우선순위: ① 방금 확정된 답변(final_answer — 복합 질문의
    후처리 실행) ② 대화 이력의 마지막 assistant 답변(단독 요청).
    """
    answer_text = str(state.get("final_answer") or "").strip()
    from_current_answer = bool(answer_text)  # 복합 실행: 방금 확정된 답변을 내보냄
    if not answer_text:
        answer_text = _last_assistant_answer(state.get("conversation_history") or [])
    if not answer_text:
        return tool_answer(NO_CONTENT_ANSWER)

    try:
        file_info = save_export_document(
            title="MARS 답변 문서",
            body=answer_text,
            filename_prefix="MARS_답변",
            tool="HWP_EXPORT",
        )
    except Exception:
        logger.exception("HWPX 생성 실패")
        return tool_answer(EXPORT_FAIL_ANSWER)

    intro = "위 답변을" if from_current_answer else "직전 답변을"
    final_answer = (
        f"{intro} 한글 문서(HWPX)로 만들었습니다.\n\n"
        f"- 파일명: {file_info['name']}\n"
        "한글 2014 이상에서 열 수 있습니다."
    )
    # file_info는 SSE file 이벤트 재료 (미들웨어가 텍스트 파싱 없이 감지)
    return tool_answer(final_answer, generated_files=[file_info])

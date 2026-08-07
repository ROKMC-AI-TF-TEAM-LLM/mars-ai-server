"""hwp_draft 도구 노드: 사용자가 제공한 내용으로 문서 초안을 작성해 HWPX로 저장.

- 초안 작성은 LLM 생성이지만 verify 밖 경로다 — 사용자가 준 내용만
  재구성하고, 없는 사실·수치·규정은 지어내지 않고 [빈칸]으로 남기도록
  프롬프트로 강제한다 (SMALLTALK과 같은 원칙: 근거 없는 창작 금지)
- HWP_EXPORT(기존 답변 그대로 저장, LLM 0회)와 역할이 다르다:
  이 도구는 "공문 초본 잡아서 파일로 만들어줘"처럼 새 문서를 만든다
- 루프 안에서 즉시 실행되는 도구다 (inline). 산출물은 EXPORT_DIR + GET /files/{파일명}
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ax_rag.query_graph.budget import trim_history
from ax_rag.query_graph.prompts import HWP_DRAFT_SYSTEM_PROMPT, history_to_messages
from ax_rag.query_graph.state import QueryState
from ax_rag.query_graph.tool_contract import tool_answer
from ax_rag.shared.config import get_config
from ax_rag.shared.exports import save_export_document
from ax_rag.shared.llm_client import get_llm
from ax_rag.shared.logging_setup import get_logger

logger = get_logger(__name__)

DRAFT_FAIL_ANSWER = "문서 초안 작성에 실패했습니다. 잠시 후 다시 시도해 주세요."


def hwp_draft(state: QueryState) -> dict:
    """사용자 제공 내용 → LLM 초안 → HWPX 저장 → 미리보기 + 다운로드 링크 응답."""
    config = get_config()
    history = trim_history(state.get("conversation_history") or [], config.HISTORY_MAX_TOKENS)
    try:
        response = (
            get_llm()
            .bind(temperature=config.GENERATE_TEMPERATURE)
            .invoke(
                [
                    SystemMessage(HWP_DRAFT_SYSTEM_PROMPT),
                    *history_to_messages(history),
                    HumanMessage(state["question"]),
                ]
            )
        )
        draft = str(response.content).strip()
    except Exception:
        logger.exception("초안 생성 실패")
        return tool_answer(DRAFT_FAIL_ANSWER)

    if not draft:
        return tool_answer(DRAFT_FAIL_ANSWER)

    try:
        file_info = save_export_document(
            title="문서 초안", body=draft, filename_prefix="MARS_초안", tool="HWP_DRAFT"
        )
    except Exception:
        logger.exception("초안 HWPX 저장 실패")
        return tool_answer(DRAFT_FAIL_ANSWER)

    final_answer = (
        "요청하신 내용으로 문서 초안을 작성해 한글 문서(HWPX)로 저장했습니다.\n\n"
        "--- 초안 미리보기 ---\n"
        f"{draft}\n"
        "---\n\n"
        f"- 파일명: {file_info['name']}\n"
        "초안이므로 [빈칸] 등 누락된 정보를 확인·보완한 뒤 사용해 주세요."
    )
    # LLM 생성 초안은 문서 근거를 주장하지 않는다 (sources 미노출).
    # file_info는 SSE file 이벤트 재료 (미들웨어가 텍스트 파싱 없이 감지)
    return tool_answer(final_answer, generated_files=[file_info])

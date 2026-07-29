"""사내 인트라넷 연동용 FastAPI 진입점 (포트 9000, interfaces.md §5).

미들웨어와의 계약:
- 요청은 JSON(question, user_department, messages[human|ai])
- 응답은 SSE(text/event-stream): text 이벤트 N회 → sources 1회 → {"type": "done"}
- 토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송이다 (architecture.md §8)

★ 단일 워커 강제: Milvus Lite 파일 락 충돌 때문에 uvicorn 멀티 워커 금지.
  실행: uvicorn main:app --host 0.0.0.0 --port 9000   (--workers 옵션 사용 금지)

구현은 ax_rag.api 패키지에 있다 (엔드포인트는 routers/, 순수 로직은
normalize·sse·pipeline). 이 파일은 앱 인스턴스 생성과 실행만 담당한다.
"""

from __future__ import annotations

from ax_rag.api import create_app
from ax_rag.api.normalize import (
    normalize_requested_domain,
    normalize_tool,
    to_internal_history,
    validate_upload,
)
from ax_rag.api.pipeline import _build_sources
from ax_rag.api.routers.documents import list_documents
from ax_rag.api.routers.files import download_file
from ax_rag.api.routers.query import capabilities
from ax_rag.api.sse import split_for_stream, sse_event, stream_answer

# 진행 상태 안내는 그래프 쪽 소유다 (API가 노드 이름을 문자열로 알지 않게)
from ax_rag.query_graph.stages import status_after_node as _status_after_node
from ax_rag.shared import vectorstore
from ax_rag.shared.logging_setup import setup_logging

# uvicorn으로 실행돼도 통일 포맷(시각 | 레벨 | 모듈 | 메시지)이 적용되게 임포트 시점에 설정
setup_logging()

app = create_app()

# 하위 호환 재수출: 기존 `import main` 후 main.<이름>으로 접근하던 코드
# (유닛 테스트 등)가 그대로 동작하도록 유지한다. 새 코드는 ax_rag.api의
# 실제 모듈에서 직접 임포트할 것
__all__ = [
    "app",
    "capabilities",
    "download_file",
    "list_documents",
    "normalize_requested_domain",
    "normalize_tool",
    "split_for_stream",
    "sse_event",
    "stream_answer",
    "to_internal_history",
    "validate_upload",
    "vectorstore",
    "_build_sources",
    "_status_after_node",
]


if __name__ == "__main__":
    import uvicorn

    # 단일 워커 강제 (Milvus Lite 파일 락). workers 인자를 절대 늘리지 말 것
    uvicorn.run(app, host="0.0.0.0", port=9000)

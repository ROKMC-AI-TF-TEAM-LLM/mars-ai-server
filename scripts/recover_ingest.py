"""미완료 갱신 저널 복구. API·다른 적재 프로세스를 중지하고 같은 설정으로 실행한다."""

from ax_rag.indexer_graph.ingest import recover_pending_ingest
from ax_rag.shared.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    print("기존 문서 복구 완료" if recover_pending_ingest() else "복구할 저널 없음")

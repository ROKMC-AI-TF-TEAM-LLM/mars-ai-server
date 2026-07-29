"""FastAPI 앱 조립 (포트 9000, interfaces.md §5).

미들웨어와의 계약:
- 요청은 JSON(question, user_department, messages[human|ai])
- 응답은 SSE(text/event-stream): text 이벤트 N회 → sources 1회 → {"type": "done"}
- 토큰 실시간 스트리밍이 아니라 verify 통과 후 분할 전송이다 (architecture.md §8)

엔드포인트 구현은 routers/ 아래 태그별 모듈에 있다. 이 파일은 앱 인스턴스
구성(제목·설명·태그·CORS)과 라우터 등록만 한다.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ax_rag.api.routers import documents, files, ops, query

# Swagger UI 그룹핑용 태그 (목적별 분류). 각 엔드포인트의 tags=와 이름을 맞춘다
_OPENAPI_TAGS = [
    {
        "name": "질의응답",
        "description": "사용자 질문 처리(SSE 스트리밍)와 요청에 넣을 수 있는 "
        "domain·tool 값 조회. 미들웨어의 채팅 경로.",
    },
    {
        "name": "문서 관리",
        "description": "RAG 문서 적재·갱신·목록·삭제와 적재 작업 상태. 관리자 "
        "페이지 데이터 소스(미들웨어가 관리자 권한 확인 후 프록시). "
        "상세 연동은 docs/middleware_document_ingest.md 참조.",
    },
    {
        "name": "생성 파일",
        "description": "도구(HWP_EXPORT 등)가 만든 문서 파일 다운로드. 미들웨어가 "
        "SSE file 이벤트를 신호로 가져가 자체 저장(fetch-and-store).",
    },
    {
        "name": "운영",
        "description": "서버·의존 서비스 상태 점검 (모니터링용).",
    },
]

# Swagger UI 최상단 설명 (미들웨어 개발자가 처음 읽는 안내)
_API_DESCRIPTION = (
    "군 내부 문서 검색 챗봇 MARS API (미들웨어 연동용).\n\n"
    "- 응답은 **SSE(text/event-stream) 스트리밍** — 이벤트 계약은 `POST /query` 문서 참조\n"
    "- 로컬 서비스 4종(vLLM :8000, 임베딩 :8001, 리랭커 :8002, Milvus)이 기동되어 있어야 한다\n"
    "- 상세 스펙: docs/interfaces.md §5"
)


def create_app() -> FastAPI:
    """MARS API 앱을 조립해 반환한다 (main.py의 uvicorn 진입점이 사용)."""
    app = FastAPI(
        title="A.X RAG 서버",
        version="0.1.0",
        description=_API_DESCRIPTION,
        openapi_tags=_OPENAPI_TAGS,
    )

    # 브라우저 프론트가 직접 붙는 개발/데모용 CORS 허용.
    # 운영 경로는 미들웨어의 서버 간 호출이라 CORS가 관여하지 않는다 (내부망 신뢰)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ops.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(files.router)
    return app

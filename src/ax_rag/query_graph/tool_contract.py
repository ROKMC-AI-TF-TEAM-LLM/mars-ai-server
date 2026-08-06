"""도구 노드의 반환 계약 (code_guide.md §12 패턴 B).

도구 노드는 검색·검증 파이프라인 **밖**에서 답을 만든다. 그래서 문서 근거를
주장하면 안 된다 — grounded=False와 빈 retrieved_chunks를 함께 돌려줘야
main.py가 sources를 붙이지 않는다. 이 세 키를 손으로 적다가 하나라도 빠지면
검증 실패 답변에 검색 상위 문서가 출처로 노출되는 사고가 난다. 그래서
헬퍼 하나로 강제한다.

이 모듈은 query_graph 안의 어떤 모듈도 임포트하지 않는다:
tools.py가 도구 노드들을 임포트하므로, 도구 노드가 tools.py를 임포트하면
순환이 된다. 계약 헬퍼를 여기 따로 두어 그 방향을 끊는다.
"""

from __future__ import annotations


def tool_answer(answer: str, generated_files: list[dict] | None = None) -> dict:
    """도구 노드의 표준 반환 dict를 만든다.

    - answer: 사용자에게 보일 도구 답변 (final_answer)
    - generated_files: 도구가 만든 파일 정보 [{"name", "url", "tool"}].
      값이 있을 때만 키를 넣는다 (파일을 만들지 않는 도구의 반환 형태 보존)

    grounded=False, retrieved_chunks=[]는 항상 함께 나간다 — 문서 근거를
    주장하지 않는다는 뜻이며, 출처 미노출의 근거가 된다.
    """
    result: dict = {"final_answer": answer, "grounded": False, "retrieved_chunks": []}
    if generated_files:
        result["generated_files"] = generated_files
    return result

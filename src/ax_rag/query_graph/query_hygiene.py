"""검색 쿼리 위생 처리 — 라우터와 ReAct 에이전트가 공유한다.

원래 nodes/router.py 안에 있던 규칙을 두 경로가 함께 쓸 수 있게 옮긴 것이다.
동작은 그대로이며, 라우터는 기존 이름으로 계속 임포트한다.

실측 근거: 재작성 쿼리에 "문서로 만들어줘" 같은 파일 요청 표현이 남으면
리랭크 최고점이 0.738 → 0.022로 무너져 검색이 전멸한다. 프롬프트 지시만으로는
작은 모델이 요청 표현을 못 떼는 경우가 있어 코드로 보강한다.
"""

from __future__ import annotations

import re

# 결정적 매처 단독 종결 기준: 이 길이 이하의 질문은 복합일 가능성이 낮아
# 매처 히트 시 LLM 없이 도구 단독으로 종결한다 (LLM 0회 이점 유지)
MATCHER_ONLY_MAX_CHARS = 30

# 검색 동반 신호: 짧은 질문이라도 이 표현이 섞이면 매처 단독으로 끝내지 않는다 —
# "해병대 주임무 찾아서 한글 파일로 저장해줘"(28자)가 검색 없이 저장만 실행된
# 사고 실측. "조사·정리·요약해서 문서로" 류의 검색 선행 표현을 폭넓게 포함한다
SEARCH_HINT_RE = re.compile(
    r"찾아|검색|알아보|알려주|조사|조회|정리|요약|참고|바탕으로|근거로|기반으로"
)

# 파일 생성 요청 표현
_FILE_REQUEST_RE = re.compile(
    r"(이\s*답변\s*[을를]?\s*)?(한글\s*)?(문서|파일)\s*(로|[을를])?\s*"
    r"(만들|생성|저장|내보내|출력|뽑|변환)\w*|문서화\s*해?\w*"
)
# 파일 표현 제거 후 끝에 남는 검색 동사 꼬리("...을 조사하여")도 정리한다
_TRAILING_SEARCH_VERB_RE = re.compile(r"[을를]?\s*(조사|조회|검색|정리|요약|알아보|찾아)\w*\s*$")


def has_search_hint(question: str) -> bool:
    """검색을 함께 요구하는 표현이 섞여 있는지 (매처 단독 종결 차단용)."""
    return bool(SEARCH_HINT_RE.search(question))


def strip_file_phrases(query: str) -> str:
    """검색 쿼리에서 파일 생성 요청 표현과 꼬리 동사를 제거한다.

    제거 결과가 너무 짧으면(검색어 실종) 원본을 유지한다 — 오염된 쿼리라도
    없는 것보다는 낫다.
    """
    cleaned = _FILE_REQUEST_RE.sub(" ", query)
    if cleaned == query:
        return query  # 파일 요청 표현이 없으면 손대지 않는다 (순수 검색어 보존)
    # 파일 표현을 걷어낸 자리에 남은 검색 동사 꼬리("…을 조사하여")를 정리한다
    cleaned = _TRAILING_SEARCH_VERB_RE.sub(" ", cleaned)
    # 끝에 남은 접속 어미·조사만 정리 (명사 일부를 깎지 않게 정확 일치로)
    cleaned = re.sub(r"(하고|하여|해서|[을를])\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.~")
    return cleaned if len(cleaned) >= 2 else query

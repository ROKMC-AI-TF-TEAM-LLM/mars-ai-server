# 구독 LLM 전환 (LiteLLM) — 작업 로그

> 브랜치 `test/litellm-subscription` (dev 기준) · 2026-10-22 · **커밋 안 함**
> 개발 노트북 전용. L40/내부망 반입 금지.

## 한 줄 요약
8000 포트에 llama.cpp 대신 LiteLLM 프록시를 띄우면 MARS가 ChatGPT 구독 모델 `gpt-6-luna`를 쓴다.
**`src/` 코드 0줄, `.env` 0줄 수정.**

## 전환 방법
| 모드 | 8000 포트에 띄울 것 |
|---|---|
| 로컬 (A.X) | `powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1` |
| 구독 (luna) | `powershell -ExecutionPolicy Bypass -File serving\start_llm_sub.ps1` |

- 켜져 있는 쪽을 Ctrl+C로 끄고 다른 쪽을 띄우면 끝. MARS(9000) 재시작 불필요
- 원리: 두 서버 모두 모델명을 `skt/A.X-4.0-Light`로 노출 → `.env`가 그대로 맞음

## 변경 파일 (전부 신규, 기존 파일 수정 없음)
| 파일 | 내용 |
|---|---|
| `serving/start_llm_sub.ps1` | 실행 스크립트 (포트 점유 검사, 외부 호출 차단용 환경변수, 경고 출력) |
| `scripts/setup_litellm_sub.ps1` | 설치 스크립트 (venv 생성, 버전 고정 설치, 버전 일치와 .pth 점검). `dev_setup.ps1`은 수정 안 함 |
| `serving/litellm_sub.yaml` | LiteLLM 모델 매핑 (alias → `chatgpt/responses/gpt-6-luna`) |
| `docs/cclog/litellm_sub.md` | 이 문서 |
| `tools/litellm-venv/` | LiteLLM 전용 venv (gitignore 대상, 저장소에 안 올라감) |

## 최초 1회 셋업
1. 설치: `powershell -ExecutionPolicy Bypass -File scripts\setup_litellm_sub.ps1` → `tools/litellm-venv`에 `litellm[proxy]==1.104.2`
   - `tools/`는 gitignore 대상이라 새로 clone하면 다시 실행해야 함 (재설치는 `-Reinstall`)
   - 프로젝트 `.venv`와 `requirements.txt`는 건드리지 않음 (의존성 충돌·규칙 3 회피)
   - 1.104.2 선택 이유: 2주 이상 지난 안정판. 1.82.7과 1.82.8은 악성 배포본(2026-03)
2. `start_llm_sub.ps1` 실행 → MARS에서 질의 1회
3. 프록시 창에 **기기 코드**가 뜸 (실측: 첫 요청 전, 기동 시점에 바로 뜸) → 브라우저에서 입력하고 ChatGPT 로그인
   - 토큰 저장 위치: `%USERPROFILE%\.config\litellm\chatgpt\auth.json` (저장소 밖)
   - 계정 보안 설정에서 "기기 코드 로그인"이 허용돼 있어야 함

## 판단 기록 (위임받아 결정한 것)
| 결정 | 이유 |
|---|---|
| 포트를 8000으로 하고 alias 방식 사용 | 기존 `start_llm_dev.ps1` 관례와 같고 전환이 가장 빠름. `config.py`의 localhost 검증을 수정 없이 통과 |
| `responses/` 접두사 사용 | 1.104.2 모델표에 `gpt-6-luna`가 없음 → 접두사 없이는 Responses 브리지를 안 탐 |
| `reasoning_effort: none` | 비추론 모드. 지연과 구독 사용량을 최소화하고 함수 호출 조건을 맞춤 |
| `drop_params: true` | temperature 등 미지원 파라미터를 에러 대신 버림 |
| `CHATGPT_DEFAULT_INSTRUCTIONS`를 중립 문구로 교체 | LiteLLM이 Codex CLI 지시문(코딩 에이전트, CLI 서식 규칙)을 모든 요청 앞에 붙임 → MARS 답변 형식이 오염됨 |
| `LITELLM_LOCAL_MODEL_COST_MAP=True` | 시작할 때 GitHub에서 가격표를 받는 외부 호출을 차단 |
| `--host 127.0.0.1` | 다른 기기에서 프록시에 접근 못 하게 함 |
| 마스터 키 없이 실행 (`LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true`) | 1차 실행 때 "no master key"로 기동 거부됨 → 수정. 키를 걸면 `.env`의 `AX_API_KEY`를 바꿔야 해서 전환 방식이 깨짐. 127.0.0.1 전용이라 노출 범위는 llama.cpp와 같음 |

## 알려진 차이 (코드 분석 기반, 실측 전)
구독 백엔드에 전달되는 필드는 `model, input, instructions, tools, tool_choice, reasoning, stream, store, include`뿐이다 (litellm `llms/chatgpt/responses/transformation.py`).

| 버려지는 것 | MARS 영향 |
|---|---|
| `temperature` | 라우터와 verify의 temperature 0 재현성이 사라짐 |
| `max_tokens` | `_STRUCTURED_MAX_TOKENS` 폭주 차단이 무력화됨 |
| `response_format`(json_schema) | 구조화 1차 경로가 문법 강제 없이 돌아감. 본문 JSON 파싱에 실패하면 tool-call 폴백으로 넘어가 호출이 1회 늘어남. **fail-closed는 유지** |

- 로그에서 `json_schema 응답이 스키마에 안 맞는다 → tool-call 시도`가 자주 보이면 위 원인 때문
- 결과는 로직과 흐름 확인용으로만 사용. 프롬프트 품질 판단이나 `experiments.md` 기록에 쓰지 말 것 (최종 검증은 L40의 A.X)

## 미검증 / 확인 필요
- [ ] 첫 실행과 로그인 (자동 모드 분류기가 프록시 실행을 막아서 사용자가 직접 실행해야 함)
- [ ] 백엔드가 교체된 지시문을 받아주는지 → 거부하면 `start_llm_sub.ps1 -CodexInstructions`로 실행
- [ ] `/health?deep=true`의 llm 항목이 ok로 나오는지 (프록시의 `/v1/models` 사용)
- [ ] 구독 플랜에서 `gpt-6-luna`가 열려 있는지 (안 열려 있으면 `... not supported when using Codex with a ChatGPT account` 에러)

## 되돌리기
`start_llm_dev.ps1`로 복귀하면 즉시 원상태. 완전히 제거하려면 다음 세 가지를 삭제:
`serving/start_llm_sub.ps1`, `serving/litellm_sub.yaml`, `tools/litellm-venv/`

## 주의
- 검색된 청크가 OpenAI로 전송됨 → **비민감 테스트 문서만** 사용
- CLAUDE.md 에어갭 규칙의 예외이며, 개발 노트북 테스트 한정

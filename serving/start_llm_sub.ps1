# =====================================================================
# 개발 노트북 전용: ChatGPT 구독 모델(gpt-6-luna)을 LiteLLM 프록시로
# 8000 포트에 OpenAI 호환으로 서빙한다. start_llm_dev.ps1(llama.cpp)의 대체품.
#
# 전환: 8000 포트에 뭘 띄우느냐만 바꾼다. .env·코드·MARS 재시작 불필요
#   로컬  : serving\start_llm_dev.ps1
#   구독  : serving\start_llm_sub.ps1   (이 스크립트)
#
# ★ 외부(chatgpt.com, auth.openai.com)로 나간다. L40/내부망 반입 금지,
#   비민감 테스트 문서만 사용. 자세한 내용은 docs/cclog/litellm_sub.md
#
# 최초 1회: 첫 요청 시 이 창에 기기 코드가 뜬다 → 브라우저에서 입력·로그인
#   토큰은 %USERPROFILE%\.config\litellm\chatgpt\auth.json (저장소 밖)
#
# 설치(1회, 프로젝트 .venv와 분리 — requirements.txt 무관):
#   py -3.11 -m venv tools\litellm-venv
#   tools\litellm-venv\Scripts\python.exe -m pip install "litellm[proxy]==1.104.2"
#   (★ 1.82.7/1.82.8은 악성 배포본. 버전 고정 유지)
#
# 사용: powershell -ExecutionPolicy Bypass -File serving\start_llm_sub.ps1
# =====================================================================
param(
    # Codex CLI 기본 지시문(코딩 에이전트 페르소나)을 그대로 쓴다.
    # 기본은 중립 문구로 교체 — 백엔드가 교체 지시문을 거부할 때만 켤 것
    [switch]$CodexInstructions,
    [int]$Port = 8000
)

$root = Split-Path $PSScriptRoot -Parent
$litellm = Join-Path $root "tools\litellm-venv\Scripts\litellm.exe"
$config = Join-Path $PSScriptRoot "litellm_sub.yaml"

if (-not (Test-Path $litellm)) {
    Write-Error "litellm이 없다: $litellm (스크립트 상단 '설치' 참고)"
    exit 1
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "포트 $Port 사용 중 — llama.cpp(start_llm_dev.ps1)를 먼저 끌 것"
    exit 1
}

# 시작 시 GitHub에서 모델 가격표를 받아오지 않게 한다 (불필요한 외부 호출 차단)
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"
$env:PYTHONUTF8 = "1"
# 마스터 키 없이 기동 허용 (없으면 litellm이 기동을 거부한다).
# 127.0.0.1 바인딩이라 노출 범위는 llama.cpp(키 없음)와 같다. 키를 걸면
# .env의 AX_API_KEY를 바꿔야 해서 "포트만 바꿔 끼우는 전환"이 깨진다
$env:LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY = "true"
if (-not $CodexInstructions) {
    # litellm이 모든 요청 앞에 붙이는 Codex CLI 지시문(코딩/CLI 서식 규칙)을
    # 중립 문구로 바꾼다. MARS 시스템 프롬프트가 그대로 주도하게 하기 위함
    $env:CHATGPT_DEFAULT_INSTRUCTIONS = "Follow the system and developer instructions given below exactly."
}

Write-Host "[구독 경로] gpt-6-luna → http://127.0.0.1:$Port/v1 (alias skt/A.X-4.0-Light)" -ForegroundColor Yellow
Write-Host "★ 외부 전송 중: 비민감 문서만 사용할 것. 로컬 복귀 = Ctrl+C 후 start_llm_dev.ps1" -ForegroundColor Yellow

& $litellm --config $config --host 127.0.0.1 --port $Port

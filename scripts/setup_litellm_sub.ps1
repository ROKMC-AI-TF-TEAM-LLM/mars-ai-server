# =====================================================================
# 개발 노트북 전용: 구독 LLM 경로(LiteLLM 프록시) 설치 — 최초 1회 실행
#
# tools/는 gitignore 대상이라 clone 직후엔 비어 있다. llama.cpp는
# dev_setup.ps1 [4/4]가 받고, LiteLLM은 이 스크립트가 받는다.
#
# 하는 일 (이미 있으면 건너뜀 / 여러 번 실행해도 안전):
#   1. tools\litellm-venv 생성 (프로젝트 .venv와 분리 — requirements.txt 무관)
#   2. litellm[proxy] 버전 고정 설치
#   3. 설치 검증: 버전 일치 + site-packages의 .pth 파일 점검 (공급망 방어)
#
# ★ 1.82.7/1.82.8은 악성 배포본(2026-03, .pth로 자격증명 탈취). 버전 고정 유지
# ★ L40/내부망 반입 금지. 실행·전환은 serving\start_llm_sub.ps1, 상세는 docs/cclog/litellm_sub.md
#
# 사전 요구: Python 3.11, 인터넷 연결
# 사용: powershell -ExecutionPolicy Bypass -File scripts\setup_litellm_sub.ps1
#       -Reinstall : venv를 지우고 새로 설치 (버전 변경 시)
# =====================================================================
param(
    [switch]$Reinstall
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# 검증된 버전으로 고정 (배포 2주 이상 경과분만 채택)
$version = "1.104.2"
$venv = "tools\litellm-venv"
$py = "$venv\Scripts\python.exe"

Write-Host "== [1/3] LiteLLM 전용 가상환경 ==" -ForegroundColor Cyan
if ($Reinstall -and (Test-Path $venv)) {
    Remove-Item -Recurse -Force $venv
    Write-Host "-Reinstall 지정 → 기존 venv 삭제"
}
if (-not (Test-Path $py)) {
    # py 런처로 3.11을 우선 찾고, 없으면 PATH의 python을 쓴다 (dev_setup.ps1과 같은 요구)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venv
    } else {
        Write-Host "Python 3.11이 없다 (py 런처/python 모두 PATH에 없음)" -ForegroundColor Red
        exit 1
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "venv 생성 실패" -ForegroundColor Red; exit 1 }
    Write-Host "venv 생성: $venv"
} else { Write-Host "venv 이미 존재 → 유지" }

Write-Host "== [2/3] litellm[proxy]==$version 설치 ==" -ForegroundColor Cyan
& $py -m pip install --quiet --disable-pip-version-check "litellm[proxy]==$version"
if ($LASTEXITCODE -ne 0) { Write-Host "pip 설치 실패 (네트워크/프록시 확인)" -ForegroundColor Red; exit 1 }

Write-Host "== [3/3] 설치 검증 ==" -ForegroundColor Cyan
$installed = & $py -c "import importlib.metadata as m; print(m.version('litellm'))"
if ($installed -ne $version) {
    Write-Host "버전 불일치: 기대 $version, 설치 $installed" -ForegroundColor Red
    exit 1
}
# .pth는 인터프리터 기동 시 자동 실행된다 (2026-03 악성 배포본의 침투 경로).
# 표준 패키지가 만드는 것 외의 .pth가 있으면 사용 전에 내용을 확인할 것
$knownPth = @("distutils-precedence.pth", "pywin32.pth")
$unknownPth = Get-ChildItem "$venv\Lib\site-packages" -Filter *.pth |
    Where-Object { $knownPth -notcontains $_.Name }
if ($unknownPth) {
    Write-Host "알 수 없는 .pth 파일 발견 — 내용 확인 전 사용 금지:" -ForegroundColor Red
    $unknownPth | ForEach-Object { Write-Host "  $($_.FullName)" }
    exit 1
}
Write-Host "litellm $installed 설치 확인, .pth 점검 통과"

Write-Host ""
Write-Host "== 설치 완료 ==" -ForegroundColor Green
Write-Host "  구독 LLM 기동 : powershell -ExecutionPolicy Bypass -File serving\start_llm_sub.ps1"
Write-Host "  로컬 LLM 복귀 : powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1"
Write-Host "  최초 기동 시 창에 기기 코드가 뜬다 → https://auth.openai.com/codex/device 에서 입력"

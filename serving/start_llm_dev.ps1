# =====================================================================
# 개발 노트북 전용 (Windows + RTX 4050): llama.cpp로 A.X 4.0 Light GGUF를
# vLLM 대신 8000 포트에 OpenAI 호환으로 서빙한다.
#
# - L40 운영 서빙은 serving/start_vllm.sh (이 스크립트는 프롬프트/로직 검증용)
# - --alias를 AX_MODEL_NAME 기본값과 맞춰 .env 수정 없이 그대로 붙는다
# - --jinja: 모델 내장 chat template 사용 → hermes 계열 tool-calling 지원
#   (주의: vLLM 파서와 동작이 다를 수 있음. tool-calling 최종 검증은 L40에서)
#
# 사용: powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1
#       (기본 실행 정책(Restricted)에서는 -ExecutionPolicy Bypass 없이 실행되지 않는다)
# =====================================================================
param(
    # RTX 4050 6GB 기준 안전값. VRAM 여유가 확인되면 99까지 올려 전체 오프로드
    [int]$NGpuLayers = 20,
    # L40 vLLM --max-model-len과 동일하게 유지 (토큰 예산 검증 일관성)
    [int]$CtxSize = 12288
)

$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root "tools\llama.cpp\llama-server.exe"
$model = Join-Path $root "models\A.X-4.0-Light-Q4_K_M.gguf"

if (-not (Test-Path $server)) {
    Write-Error "llama-server.exe가 없다: $server (tools/llama.cpp에 릴리스 바이너리를 풀어둘 것)"
    exit 1
}
if (-not (Test-Path $model)) {
    Write-Error "GGUF 모델이 없다: $model"
    exit 1
}

& $server `
    -m $model `
    --host 127.0.0.1 `
    --port 8000 `
    --alias "skt/A.X-4.0-Light" `
    --jinja `
    --ctx-size $CtxSize `
    -ngl $NGpuLayers

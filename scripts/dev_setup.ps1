# =====================================================================
# 개발 노트북(Windows) 환경 부트스트랩 — git clone 직후 1회 실행
#
# 하는 일 (이미 있으면 건너뜀 / 여러 번 실행해도 안전):
#   0. 사전 요구 검사 (Python 3.11, 디스크) — 다운로드 전에 즉시 중단
#   1. .venv 생성 + 개발/실행 의존성 설치 (버전 고정)
#   2. .env 생성 (.env.dev.example 복사)
#   3. 모델 다운로드: A.X GGUF(4.1GB), bge-m3, bge-reranker-v2-m3 (~9GB)
#   4. llama.cpp 릴리스 바이너리 다운로드 (tools/llama.cpp)
#
# ★ Docker가 더 이상 필요 없다. milvus-lite 3.x가 순수 파이썬이라 Windows에서도
#   임베디드로 뜬다 — 운영 L40과 **같은 엔진**이다 (docs/milvus_lite_3x.md).
#   벡터DB는 .env의 MILVUS_LITE_PATH가 가리키는 로컬 경로에 자동 생성된다.
#
# 사전 요구: Python 3.11, git, 인터넷 연결
# 사용: powershell -ExecutionPolicy Bypass -File scripts\dev_setup.ps1
#       ★ Windows 기본 실행 정책(Restricted)에서는 -ExecutionPolicy Bypass 없이
#         "running scripts is disabled" 오류로 실행되지 않는다
#       (모델 폴더를 백업해 뒀다면 models/, tools/ 복원 후 실행 → 다운로드 생략)
# =====================================================================
param(
    [switch]$SkipModels   # 모델 다운로드 건너뛰기 (백업 복원 시)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host "== [0/4] 사전 요구 검사 ==" -ForegroundColor Cyan
# Python: PATH 존재부터 확인한다 (없으면 (python --version)이 알 수 없는 오류로 죽는다)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "python이 PATH에 없다. Python 3.11을 설치할 것 (requires-python==3.11.*)" -ForegroundColor Red
    exit 1
}
$pyVersion = & python --version   # 3.4+는 버전을 stdout으로 출력
if ($pyVersion -notmatch "3\.11") {
    Write-Warning "Python 3.11이 아니다: $pyVersion (requires-python==3.11.*)"
}
$driveName = (Split-Path $root -Qualifier).TrimEnd(":")
$freeGB = [math]::Round((Get-PSDrive -Name $driveName).Free / 1GB, 1)
if (-not $SkipModels -and $freeGB -lt 15) {
    Write-Warning "디스크 여유가 ${freeGB}GB뿐이다. 모델(~9GB)+도구 다운로드에 15GB 이상 권장"
}
Write-Host "사전 검사 통과 (python=$pyVersion, 디스크 여유 ${freeGB}GB)"

Write-Host "== [1/4] 파이썬 가상환경 + 의존성 ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
# setuptools<81 고정: pytest와 sdist 빌드(FlagEmbedding)가 pkg_resources를 쓰는데
# 81+에서 제거됐다 (pymilvus는 3.x부터 쓰지 않아 더 이상 이유가 아니다)
& $py -m pip install --quiet "setuptools==75.6.0"
if ($LASTEXITCODE -ne 0) { Write-Host "pip 설치 실패: setuptools (네트워크/프록시 확인)" -ForegroundColor Red; exit 1 }
& $py -m pip install --quiet `
    pytest==8.3.4 ruff==0.8.6 python-dotenv==1.0.1 `
    fastapi==0.115.6 pydantic==2.10.4 requests==2.32.3 "uvicorn[standard]==0.34.0" `
    langgraph==0.2.62 langchain-core==0.3.29 langchain-openai==0.2.14 langchain-text-splitters==0.3.4 `
    pymilvus==3.0.1 milvus-lite==3.2.1 faiss-cpu==1.15.0 `
    kiwipiepy==0.22.2 bm25s==0.2.5 pdfplumber==0.11.10
if ($LASTEXITCODE -ne 0) { Write-Host "pip 설치 실패: 앱 의존성 (네트워크/프록시 확인)" -ForegroundColor Red; exit 1 }
# FlagEmbedding은 transformers 상한이 낮아 별도 설치 (vllm과 같은 venv 불가 — 노트북엔 vllm 없음)
& $py -m pip install --quiet torch==2.8.0 FlagEmbedding==1.3.3
if ($LASTEXITCODE -ne 0) { Write-Host "pip 설치 실패: torch/FlagEmbedding" -ForegroundColor Red; exit 1 }
Write-Host "의존성 설치 완료"

Write-Host "== [2/4] .env ==" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.dev.example" ".env"
    Write-Host ".env 생성 (.env.dev.example 복사)"
} else {
    Write-Host ".env 이미 존재 → 유지"
}

Write-Host "== [3/4] 모델 다운로드 ==" -ForegroundColor Cyan
if ($SkipModels) {
    Write-Host "-SkipModels 지정 → 건너뜀"
} else {
    New-Item -ItemType Directory -Force models | Out-Null
    $gguf = "models\A.X-4.0-Light-Q4_K_M.gguf"
    if (-not (Test-Path $gguf)) {
        Write-Host "A.X GGUF 다운로드 중 (~4.1GB)..."
        curl.exe -L --ssl-no-revoke --fail -o "$gguf.part" `
            "https://huggingface.co/mykor/A.X-4.0-Light-gguf/resolve/main/A.X-4.0-Light-Q4_K_M.gguf"
        if ($LASTEXITCODE -ne 0) { Write-Host "GGUF 다운로드 실패 (네트워크/URL 확인)" -ForegroundColor Red; exit 1 }
        Move-Item "$gguf.part" $gguf -Force
    } else { Write-Host "GGUF 이미 존재 → 건너뜀" }

    foreach ($m in @(
        @{repo = "BAAI/bge-m3"; dir = "models/bge-m3"},
        @{repo = "BAAI/bge-reranker-v2-m3"; dir = "models/bge-reranker-v2-m3"}
    )) {
        if (-not (Test-Path $m.dir)) {
            Write-Host "$($m.repo) 다운로드 중..."
            & $py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$($m.repo)', local_dir='$($m.dir)', ignore_patterns=['onnx/*','*.onnx','imgs/*','*.md'])"
            if ($LASTEXITCODE -ne 0) { Write-Host "$($m.repo) 다운로드 실패" -ForegroundColor Red; exit 1 }
        } else { Write-Host "$($m.dir) 이미 존재 → 건너뜀" }
    }
}

Write-Host "== [4/4] llama.cpp 바이너리 ==" -ForegroundColor Cyan
if (-not (Test-Path "tools\llama.cpp\llama-server.exe")) {
    New-Item -ItemType Directory -Force tools | Out-Null
    $tag = "b9870"  # 검증된 릴리스로 고정
    Write-Host "llama.cpp $tag (CUDA 12.4) 다운로드 중..."
    curl.exe -sS -L --ssl-no-revoke --fail -o "tools\llama-cuda.zip" `
        "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-12.4-x64.zip"
    if ($LASTEXITCODE -ne 0) { Write-Host "llama.cpp 다운로드 실패" -ForegroundColor Red; exit 1 }
    curl.exe -sS -L --ssl-no-revoke --fail -o "tools\cudart.zip" `
        "https://github.com/ggml-org/llama.cpp/releases/download/$tag/cudart-llama-bin-win-cuda-12.4-x64.zip"
    if ($LASTEXITCODE -ne 0) { Write-Host "cudart 다운로드 실패" -ForegroundColor Red; exit 1 }
    Expand-Archive tools\llama-cuda.zip -DestinationPath tools\llama.cpp -Force
    Expand-Archive tools\cudart.zip -DestinationPath tools\llama.cpp -Force
    Remove-Item tools\llama-cuda.zip, tools\cudart.zip
} else { Write-Host "llama.cpp 이미 존재 → 건너뜀" }

# 벡터DB는 별도 단계가 없다 — Milvus Lite가 첫 적재 때 .env의 MILVUS_LITE_PATH에
# 파일을 만든다. 접속되는지만 미리 확인해 셋업 단계에서 문제를 잡는다
Write-Host "== 벡터DB 확인 (Milvus Lite, Docker 불필요) ==" -ForegroundColor Cyan
& $py -c "from pymilvus import MilvusClient; import tempfile, os, shutil; d = tempfile.mkdtemp(); MilvusClient(os.path.join(d, 'probe.db')); shutil.rmtree(d, ignore_errors=True); print('Milvus Lite 접속 확인')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Milvus Lite 접속 실패 — milvus-lite/faiss-cpu 설치 상태를 확인할 것" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "== 셋업 완료. 다음 순서로 기동: ==" -ForegroundColor Green
Write-Host "  1) powershell -ExecutionPolicy Bypass -File serving\start_llm_dev.ps1   # LLM :8000"
Write-Host "  2) `$env:PYTHONPATH='src'; $py serving\embedding_server.py    # 임베딩 :8001"
Write-Host "  3) `$env:PYTHONPATH='src'; $py serving\reranker_server.py     # 리랭커 :8002"
Write-Host "  4) `$env:PYTHONPATH='src'; $py scripts\bulk_ingest.py --dir sample_docs --domain HR --department HR_TEAM --visibility ALL"
Write-Host "     ★ --domain은 문서 성격에 맞게 준다. 전부 한 도메인으로 넣으면 검색에서 배제된다"
Write-Host "  5) `$env:PYTHONPATH='src'; $py -m uvicorn main:app --host 0.0.0.0 --port 9000"
Write-Host "     ★ --workers 금지 (Milvus Lite 파일 락)"
Write-Host "  검증: $py -m pytest -q  /  curl http://localhost:9000/health"

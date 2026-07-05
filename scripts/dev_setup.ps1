# =====================================================================
# 개발 노트북(Windows) 환경 부트스트랩 — git clone 직후 1회 실행
#
# 하는 일 (이미 있으면 건너뜀 / 여러 번 실행해도 안전):
#   1. .venv 생성 + 개발/실행 의존성 설치 (버전 고정)
#   2. .env 생성 (.env.dev.example 복사)
#   3. 모델 다운로드: A.X GGUF(4.1GB), bge-m3, bge-reranker-v2-m3 (~9GB)
#   4. llama.cpp 릴리스 바이너리 다운로드 (tools/llama.cpp)
#   5. Milvus standalone Docker 컨테이너 생성 (ax-milvus-dev)
#
# 사전 요구: Python 3.11, git, Docker Desktop(실행 중), 인터넷 연결
# 사용: powershell -File scripts\dev_setup.ps1
#       (모델 폴더를 백업해 뒀다면 models/, tools/ 복원 후 실행 → 다운로드 생략)
# =====================================================================
param(
    [switch]$SkipModels   # 모델 다운로드 건너뛰기 (백업 복원 시)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host "== [1/5] 파이썬 가상환경 + 의존성 ==" -ForegroundColor Cyan
$pyVersion = (python --version) 2>&1
if ($pyVersion -notmatch "3\.11") {
    Write-Warning "Python 3.11이 아니다: $pyVersion (requires-python==3.11.*)"
}
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
# setuptools<81 고정: pymilvus 2.5.4가 pkg_resources 필요 (81+에서 제거됨)
& $py -m pip install --quiet "setuptools==75.6.0"
& $py -m pip install --quiet `
    pytest==8.3.4 ruff==0.8.6 python-dotenv==1.0.1 `
    fastapi==0.115.6 pydantic==2.10.4 requests==2.32.3 "uvicorn[standard]==0.34.0" `
    langgraph==0.2.62 langchain-core==0.3.29 langchain-openai==0.2.14 langchain-text-splitters==0.3.4 `
    pymilvus==2.5.4 kiwipiepy==0.22.2 bm25s==0.2.5
# FlagEmbedding은 transformers 상한이 낮아 별도 설치 (vllm과 같은 venv 불가 — 노트북엔 vllm 없음)
& $py -m pip install --quiet torch==2.8.0 FlagEmbedding==1.3.3
Write-Host "의존성 설치 완료"

Write-Host "== [2/5] .env ==" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.dev.example" ".env"
    Write-Host ".env 생성 (.env.dev.example 복사)"
} else {
    Write-Host ".env 이미 존재 → 유지"
}

Write-Host "== [3/5] 모델 다운로드 ==" -ForegroundColor Cyan
if ($SkipModels) {
    Write-Host "-SkipModels 지정 → 건너뜀"
} else {
    New-Item -ItemType Directory -Force models | Out-Null
    $gguf = "models\A.X-4.0-Light-Q4_K_M.gguf"
    if (-not (Test-Path $gguf)) {
        Write-Host "A.X GGUF 다운로드 중 (~4.1GB)..."
        curl.exe -L --ssl-no-revoke --fail -o "$gguf.part" `
            "https://huggingface.co/mykor/A.X-4.0-Light-gguf/resolve/main/A.X-4.0-Light-Q4_K_M.gguf"
        Move-Item "$gguf.part" $gguf -Force
    } else { Write-Host "GGUF 이미 존재 → 건너뜀" }

    foreach ($m in @(
        @{repo = "BAAI/bge-m3"; dir = "models/bge-m3"},
        @{repo = "BAAI/bge-reranker-v2-m3"; dir = "models/bge-reranker-v2-m3"}
    )) {
        if (-not (Test-Path $m.dir)) {
            Write-Host "$($m.repo) 다운로드 중..."
            & $py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$($m.repo)', local_dir='$($m.dir)', ignore_patterns=['onnx/*','*.onnx','imgs/*','*.md'])"
        } else { Write-Host "$($m.dir) 이미 존재 → 건너뜀" }
    }
}

Write-Host "== [4/5] llama.cpp 바이너리 ==" -ForegroundColor Cyan
if (-not (Test-Path "tools\llama.cpp\llama-server.exe")) {
    New-Item -ItemType Directory -Force tools | Out-Null
    $tag = "b9870"  # 검증된 릴리스로 고정
    Write-Host "llama.cpp $tag (CUDA 12.4) 다운로드 중..."
    curl.exe -sS -L --ssl-no-revoke --fail -o "tools\llama-cuda.zip" `
        "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-12.4-x64.zip"
    curl.exe -sS -L --ssl-no-revoke --fail -o "tools\cudart.zip" `
        "https://github.com/ggml-org/llama.cpp/releases/download/$tag/cudart-llama-bin-win-cuda-12.4-x64.zip"
    Expand-Archive tools\llama-cuda.zip -DestinationPath tools\llama.cpp -Force
    Expand-Archive tools\cudart.zip -DestinationPath tools\llama.cpp -Force
    Remove-Item tools\llama-cuda.zip, tools\cudart.zip
} else { Write-Host "llama.cpp 이미 존재 → 건너뜀" }

Write-Host "== [5/5] Milvus standalone (Docker) ==" -ForegroundColor Cyan
$existing = docker ps -a --filter "name=ax-milvus-dev" --format "{{.Names}}"
if ($existing -eq "ax-milvus-dev") {
    Write-Host "컨테이너 이미 존재 → docker start ax-milvus-dev"
    docker start ax-milvus-dev | Out-Null
} else {
    New-Item -ItemType Directory -Force "data\milvus-docker" | Out-Null
    Copy-Item "serving\milvus-dev\embedEtcd.yaml" "data\milvus-docker\" -Force
    Copy-Item "serving\milvus-dev\user.yaml" "data\milvus-docker\" -Force
    $dir = Join-Path $root "data\milvus-docker"
    docker run -d --name ax-milvus-dev --security-opt seccomp:unconfined `
        -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd `
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml -e COMMON_STORAGETYPE=local `
        -v "${dir}\embedEtcd.yaml:/milvus/configs/embedEtcd.yaml" `
        -v "${dir}\user.yaml:/milvus/configs/user.yaml" `
        -v "${dir}\data:/var/lib/milvus" `
        -p 19530:19530 -p 9091:9091 `
        milvusdb/milvus:v2.5.4 milvus run standalone | Out-Null
    Write-Host "컨테이너 생성 완료 (이미지 최초 pull 시 수 분 소요)"
}

Write-Host ""
Write-Host "== 셋업 완료. 다음 순서로 기동: ==" -ForegroundColor Green
Write-Host "  1) powershell -File serving\start_llm_dev.ps1                 # LLM :8000"
Write-Host "  2) `$env:PYTHONPATH='src'; $py serving\embedding_server.py    # 임베딩 :8001"
Write-Host "  3) `$env:PYTHONPATH='src'; $py serving\reranker_server.py     # 리랭커 :8002"
Write-Host "  4) `$env:PYTHONPATH='src'; $py scripts\bulk_ingest.py --dir sample_docs --domain HR --department HR_TEAM --visibility ALL"
Write-Host "  5) `$env:PYTHONPATH='src'; $py -m uvicorn main:app --host 0.0.0.0 --port 9000"
Write-Host "  검증: $py -m pytest -q  /  curl http://localhost:9000/health"

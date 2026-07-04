#!/usr/bin/env bash
# vLLM 서빙 기동 (L40, interfaces.md §11 그대로)
# /local/path/to/A.X-4.0-Light 는 오프라인 반입된 실제 로컬 경로로 교체한다.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve /local/path/to/A.X-4.0-Light \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.78 \
  --max-model-len 12288 \
  --max-num-seqs 16 \
  --port 8000

#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TASK_DIR="$(cd "$REPO_DIR/../.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$TASK_DIR/work/vllm-venv/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-$TASK_DIR/work/models/Qwen3-8B-4bit}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export VLLM_NO_USAGE_STATS=1
export HF_HUB_DISABLE_TELEMETRY=1
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name local-qwen3-8b --host 127.0.0.1 --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager

#!/usr/bin/env bash
set -euo pipefail

traits=(extraversion agreeableness conscientiousness)
model="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
vector_metadata="${VECTOR_METADATA:-configs/steering.layers.yaml}"
vector_root_default="$(VECTOR_METADATA_PATH="$vector_metadata" python3 <<'PY'
from pathlib import Path
import os
import yaml

config_path = Path(os.environ["VECTOR_METADATA_PATH"])
config = yaml.safe_load(config_path.read_text()) or {}
root = config.get("vector_root") or "data/vectors"
path = Path(root)
if not path.is_absolute():
    path = (config_path.parent / path).resolve()
print(path)
PY
)"
vector_root="${VECTOR_ROOT:-$vector_root_default}"
prompt_dir="${PROMPT_DIR:-data/prompts}"
alpha="${STEERING_ALPHA:-1.0}"
delta_threshold="${DELTA_THRESHOLD:-0.1}"
sign_threshold="${SIGN_THRESHOLD:-0.55}"
artifact_dir="${ARTIFACT_DIR:-artifacts/steering_eval}"

printf '[vector-eval] Regenerating steering vectors using %s...\n' "$vector_metadata"
VECTOR_METADATA="$vector_metadata" VECTOR_ROOT="$vector_root" ./scripts/compute_vectors.sh

mkdir -p "$artifact_dir"

printf '[vector-eval] Evaluating traits %s with model %s\n' "${traits[*]}" "$model"
python3 -m steering.eval \
  --model "$model" \
  --metadata-root "$vector_root" \
  --prompt-dir "$prompt_dir" \
  --eval-suffix "_eval" \
  --traits "${traits[@]}" \
  --alpha "$alpha" \
  --delta-threshold "$delta_threshold" \
  --sign-threshold "$sign_threshold" \
  --json-output "$artifact_dir/report.json" \
  --markdown-output "$artifact_dir/report.md"

#!/usr/bin/env bash
set -euo pipefail

traits=(${TRAITS:-extraversion agreeableness conscientiousness})
vector_metadata="${VECTOR_METADATA:-configs/steering.layers.yaml}"
model_default="$(VECTOR_METADATA_PATH="$vector_metadata" python3 <<'PY'
from pathlib import Path
import os
import yaml

config_path = Path(os.environ["VECTOR_METADATA_PATH"])
config = yaml.safe_load(config_path.read_text()) or {}
print((config.get("defaults") or {}).get("model") or "meta-llama/Llama-3.1-8B-Instruct")
PY
)"
model="${MODEL_NAME:-$model_default}"
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
alpha_grid="${ALPHA_GRID:-}"
delta_threshold="${DELTA_THRESHOLD:-0.1}"
sign_threshold="${SIGN_THRESHOLD:-0.55}"
anti_threshold="${ANTI_STEERABLE_THRESHOLD:-0.5}"
artifact_dir="${ARTIFACT_DIR:-artifacts/steering_eval}"

if [ "${SKIP_VECTOR_REGEN:-0}" != "1" ]; then
  printf '[vector-eval] Regenerating steering vectors using %s...\n' "$vector_metadata"
  VECTOR_METADATA="$vector_metadata" VECTOR_ROOT="$vector_root" ./scripts/compute_vectors.sh
else
  printf '[vector-eval] Skipping vector regeneration (SKIP_VECTOR_REGEN=%s)\n' "${SKIP_VECTOR_REGEN}"
fi

mkdir -p "$artifact_dir"

if [ "${BASELINE_BFI:-0}" = "1" ]; then
  printf '[vector-eval] Writing baseline BFI prompt skeleton to %s\n' "$artifact_dir/baseline_bfi_prompt.txt"
  python3 -m steering.baseline_bfi prompt > "$artifact_dir/baseline_bfi_prompt.txt"
fi

printf '[vector-eval] Evaluating traits %s with model %s\n' "${traits[*]}" "$model"
eval_args=(
  --model "$model"
  --metadata-root "$vector_root"
  --prompt-dir "$prompt_dir"
  --eval-suffix "_eval"
  --traits "${traits[@]}"
  --alpha "$alpha"
  --delta-threshold "$delta_threshold"
  --sign-threshold "$sign_threshold"
  --anti-steerable-threshold "$anti_threshold"
  --json-output "$artifact_dir/report.json"
  --markdown-output "$artifact_dir/report.md"
)

if [ -n "$alpha_grid" ]; then
  eval_args+=(--alpha-grid "$alpha_grid")
fi

if [ "${MEASURE_BLEED:-0}" = "1" ]; then
  eval_args+=(--measure-bleed)
fi

python3 -m steering.eval "${eval_args[@]}"

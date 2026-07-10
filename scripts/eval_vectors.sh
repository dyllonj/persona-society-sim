#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$python_bin" ]; then
  if [ -x ".venv/bin/python" ]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

traits=(${TRAITS:-extraversion agreeableness conscientiousness})
vector_metadata="${VECTOR_METADATA:-configs/steering.layers.yaml}"
model_default="$(VECTOR_METADATA_PATH="$vector_metadata" "$python_bin" <<'PY'
from pathlib import Path
import os
import yaml

config_path = Path(os.environ["VECTOR_METADATA_PATH"])
config = yaml.safe_load(config_path.read_text()) or {}
print((config.get("defaults") or {}).get("model") or "meta-llama/Llama-3.1-8B-Instruct")
PY
)"
model="${MODEL_NAME:-$model_default}"
model_revision="${MODEL_REVISION:-}"
tokenizer_revision="${TOKENIZER_REVISION:-$model_revision}"
vector_root_default="$(VECTOR_METADATA_PATH="$vector_metadata" "$python_bin" <<'PY'
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
alpha_default="$(VECTOR_METADATA_PATH="$vector_metadata" "$python_bin" <<'PY'
from pathlib import Path
import os
import yaml

config_path = Path(os.environ["VECTOR_METADATA_PATH"])
config = yaml.safe_load(config_path.read_text()) or {}
print((config.get("defaults") or {}).get("eval_alpha") or 1.0)
PY
)"
alpha="${STEERING_ALPHA:-$alpha_default}"
trait_alphas="${TRAIT_ALPHAS:-}"
alpha_grid="${ALPHA_GRID:-}"
inference_dtype="${INFERENCE_DTYPE:-bf16}"
delta_threshold="${DELTA_THRESHOLD:-0.1}"
sign_threshold="${SIGN_THRESHOLD:-0.55}"
anti_threshold="${ANTI_STEERABLE_THRESHOLD:-0.5}"
artifact_dir="${ARTIFACT_DIR:-artifacts/steering_eval}"

if [ "${OVERWRITE:-0}" != "1" ] && {
  [ -e "$artifact_dir/report.json" ] || [ -e "$artifact_dir/report.md" ];
}; then
  printf '[vector-eval] Refusing to overwrite %s/report.{json,md}; set OVERWRITE=1 deliberately.\n' "$artifact_dir" >&2
  exit 2
fi

if [ "${SKIP_VECTOR_REGEN:-0}" != "1" ]; then
  printf '[vector-eval] Regenerating steering vectors using %s...\n' "$vector_metadata"
  VECTOR_METADATA="$vector_metadata" VECTOR_ROOT="$vector_root" ./scripts/compute_vectors.sh
else
  printf '[vector-eval] Skipping vector regeneration (SKIP_VECTOR_REGEN=%s)\n' "${SKIP_VECTOR_REGEN}"
fi

mkdir -p "$artifact_dir"

if [ "${BASELINE_BFI:-0}" = "1" ]; then
  printf '[vector-eval] Writing baseline BFI prompt skeleton to %s\n' "$artifact_dir/baseline_bfi_prompt.txt"
  "$python_bin" -m steering.baseline_bfi prompt > "$artifact_dir/baseline_bfi_prompt.txt"
fi

printf '[vector-eval] Evaluating traits %s with model %s\n' "${traits[*]}" "$model"
eval_args=(
  --model "$model"
  --metadata-root "$vector_root"
  --vector-config "$vector_metadata"
  --prompt-dir "$prompt_dir"
  --eval-suffix "_eval"
  --traits "${traits[@]}"
  --alpha "$alpha"
  --dtype "$inference_dtype"
  --delta-threshold "$delta_threshold"
  --sign-threshold "$sign_threshold"
  --anti-steerable-threshold "$anti_threshold"
  --json-output "$artifact_dir/report.json"
  --markdown-output "$artifact_dir/report.md"
)

if [ -n "$trait_alphas" ]; then
  eval_args+=(--trait-alpha "$trait_alphas")
fi

if [ -n "$model_revision" ]; then
  eval_args+=(--model-revision "$model_revision")
fi
if [ -n "$tokenizer_revision" ]; then
  eval_args+=(--tokenizer-revision "$tokenizer_revision")
fi

if [ -n "$alpha_grid" ]; then
  eval_args+=(--alpha-grid "$alpha_grid")
fi

if [ "${MEASURE_BLEED:-0}" = "1" ]; then
  eval_args+=(--measure-bleed)
fi

"$python_bin" -m steering.eval "${eval_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

traits=(extraversion agreeableness conscientiousness)
model="meta-llama/Llama-3.1-8B-Instruct"
vector_root="data/vectors"

trait_code() {
  case "$1" in
    extraversion) echo "E" ;;
    agreeableness) echo "A" ;;
    conscientiousness) echo "C" ;;
    *) echo "X" ;;
  esac
}

for trait in "${traits[@]}"; do
  code=$(trait_code "$trait")
  prompt_path="data/prompts/${trait}_eval.jsonl"
  if [[ ! -f "$prompt_path" ]]; then
    echo "[layer-sweep] Missing held-out prompts for ${trait}, using training file" >&2
    prompt_path="data/prompts/${trait}.jsonl"
  fi
  python3 -m steering.layer_sweep \
    "$code" \
    "$prompt_path" \
    "$vector_root" \
    --model "$model" \
    --vector-store-id "$code"
done

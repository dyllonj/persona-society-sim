#!/usr/bin/env bash
set -euo pipefail

traits=(extraversion agreeableness conscientiousness)
model="meta-llama/Llama-3-8b-instruct"

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
  python3 -m steering.compute_caa \
    "$code" \
    "data/prompts/${trait}.jsonl" \
    data/vectors \
    --model "$model"
done

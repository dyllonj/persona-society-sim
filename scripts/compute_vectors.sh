#!/usr/bin/env bash
set -euo pipefail

config_file="${VECTOR_METADATA:-configs/steering.layers.yaml}"
model="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
vector_root_override="${VECTOR_ROOT:-}"
selected_traits="${TRAITS:-}"

python3 <<'PY' "$config_file" "$model" "$vector_root_override" "$selected_traits"
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model_name = sys.argv[2]
root_override = sys.argv[3]
selected = {item.strip() for item in sys.argv[4].split(",") if item.strip()} if sys.argv[4] else None

config = yaml.safe_load(config_path.read_text()) or {}
defaults = config.get("defaults") or {}
traits_cfg = config.get("traits") or {}
if not traits_cfg:
    raise SystemExit("No trait definitions found in steering metadata file")

vector_root = Path(root_override) if root_override else Path(config.get("vector_root") or "data/vectors")
if not vector_root.is_absolute():
    vector_root = (config_path.parent / vector_root).resolve()
vector_root.mkdir(parents=True, exist_ok=True)

prompt_dir = Path(defaults.get("prompt_dir") or "data/prompts")
if not prompt_dir.is_absolute():
    prompt_dir = (config_path.parent / prompt_dir).resolve()

default_layers = defaults.get("layers")
if not default_layers:
    raise SystemExit("defaults.layers must be provided in steering metadata")

def resolve_path(value: str, fallback_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (fallback_dir / path).resolve()
    return path

for trait_code, spec in traits_cfg.items():
    if selected and trait_code not in selected:
        continue
    prompt_file = spec.get("prompt_file")
    if prompt_file:
        prompt_path = resolve_path(prompt_file, config_path.parent)
    else:
        name = spec.get("name") or trait_code
        prompt_path = prompt_dir / f"{name.lower()}.jsonl"
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file missing for trait {trait_code}: {prompt_path}")
    layers = spec.get("layers") or default_layers
    if not layers:
        raise SystemExit(f"No layer list defined for trait {trait_code}")
    vector_store_id = spec.get("vector_store_id") or trait_code
    command = [
        sys.executable,
        "-m",
        "steering.compute_caa",
        trait_code,
        str(prompt_path),
        str(vector_root),
        "--model",
        model_name,
        "--vector-store-id",
        vector_store_id,
        "--layers",
        *[str(layer) for layer in layers],
    ]
    print(f"[compute-vectors] trait={trait_code} layers={layers} store_id={vector_store_id}")
    subprocess.run(command, check=True)
PY

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

config_file="${VECTOR_METADATA:-configs/steering.layers.yaml}"
model_override="${MODEL_NAME:-}"
vector_root_override="${VECTOR_ROOT:-}"
selected_traits="${TRAITS:-}"

"$python_bin" - "$config_file" "$model_override" "$vector_root_override" "$selected_traits" <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model_override = sys.argv[2]
root_override = sys.argv[3]
selected = {item.strip() for item in sys.argv[4].split(",") if item.strip()} if sys.argv[4] else None

config = yaml.safe_load(config_path.read_text()) or {}
traits_cfg = config.get("traits") or {}
if not traits_cfg:
    raise SystemExit("No trait definitions found in steering metadata file")

for trait_code in traits_cfg:
    if selected and trait_code not in selected:
        continue
    command = [
        sys.executable,
        "-m",
        "steering.compute_caa",
        trait_code,
        "--config",
        str(config_path),
    ]
    if model_override:
        command.extend(["--model", model_override])
    if root_override:
        root_path = Path(root_override)
        if not root_path.is_absolute():
            root_path = (config_path.parent / root_path).resolve()
        command.extend(["--output-dir", str(root_path)])
    print(f"[compute-vectors] trait={trait_code} config={config_path}")
    subprocess.run(command, check=True)
PY

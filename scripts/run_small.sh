#!/usr/bin/env bash
set -euo pipefail

python <<'PY'
import yaml
from pathlib import Path

config = yaml.safe_load(Path("configs/run.small.yaml").read_text())
print("Loaded config:", config["run_id"], "population", config["population"], "steps", config["steps"])
print("TODO: wire up SimulationRunner CLI once the agent pool is ready.")
PY

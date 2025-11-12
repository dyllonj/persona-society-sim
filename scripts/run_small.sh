#!/usr/bin/env bash
set -euo pipefail

python3 -m orchestrator.cli configs/run.small.yaml --mock-model "$@"

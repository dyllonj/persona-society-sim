"""Make a portable, publishable copy of a completed distributed-clue batch.

Only local path metadata changes: config.output becomes the batch name and each
served model root becomes its public checkpoint ID. Traces, receipts, prompts and
scores are copied unchanged, and the copy must replay to the same verification
result as the original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_distributed_clues import verify_batch

PUBLIC_CHECKPOINTS = {
    "local-qwen3-8b": "mlx-community/Qwen3-8B-4bit",
    "local-qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
}


def export(source: Path, destination: Path) -> dict:
    original = verify_batch(source)
    shutil.copytree(source, destination, dirs_exist_ok=False)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["output"] = destination.name
    for model in manifest["server_models"]["data"]:
        model["root"] = PUBLIC_CHECKPOINTS[model["id"]] + " (local download)"
        if model.get("parent"):
            model["parent"] = None
    manifest["export_note"] = ("Portable export: config.output and server_models.data[].root/parent "
                               "replace local paths. Traces, receipts, prompts and scores are unchanged.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "verification.json").write_text(json.dumps(original, indent=2) + "\n")
    leftover = [p.relative_to(destination) for p in destination.rglob("*")
                if p.is_file() and str(Path.home()) in p.read_text(errors="ignore")]
    if leftover:
        raise ValueError(f"Local home path remains in {leftover[:5]}")
    if verify_batch(destination) != original:
        raise ValueError("Exported batch no longer replays to the original verification result")
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.destination.resolve()), indent=2))


if __name__ == "__main__":
    main()

"""Run one bounded, auditable pilot in the existing town (no cloud calls)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.cli import load_config
import yaml


def summarize(db_path):
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        rows = list(db.execute("SELECT * FROM action_log ORDER BY tick, rowid"))
        messages = list(db.execute("SELECT tokens_in,tokens_out FROM msg_log"))
        grades = list(db.execute("SELECT * FROM report_grade_log"))
    source = Counter(json.loads(row["plan_metadata"] or "{}").get("decision_source", "unknown") for row in rows)
    return {
        "actions": len(rows), "decision_sources": dict(source),
        "action_types": dict(Counter(row["action_type"] for row in rows)),
        "action_outcomes": dict(Counter(row["outcome"] for row in rows)),
        "report_submissions": len(grades),
        "report_grades": [dict(row) for row in grades],
        "tokens_in": sum(row["tokens_in"] for row in messages),
        "tokens_out": sum(row["tokens_out"] for row in messages),
        "interpretation": "Single local pilot. Action success is not collaboration or objective completion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20 or not 1 <= args.population <= 6:
        parser.error("Local pilot limited to 1..20 steps and 1..6 agents")
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
        served = json.load(response)
    if "local-qwen3-8b" not in [item["id"] for item in served["data"]]:
        raise RuntimeError("Expected local-qwen3-8b server is not running")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never append to or overwrite an old episode.
    config = load_config(ROOT / "configs/run.local-8b.yaml")
    config.update(run_id=out.name, steps=args.steps, population=args.population, seed=args.seed)
    config["logging"] = {"db_url": "sqlite:///" + str(out / "simulation.db"),
                         "parquet_dir": str(out / "logs")}
    config["inference"]["receipt_dir"] = str(out / "receipts")
    config_path = out / "resolved-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    files = [ROOT / name for name in subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines() if name.endswith((".py", ".yaml", ".txt", ".toml")) and not name.startswith("artifacts/")]
    manifest = {
        "schema_version": 1, "status": "running", "real_inference": True,
        "server_models": served, "model_revision": config["model_revision"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "runtime": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "activation_steering": False, "persona_prompt": False,
        "note": "Model revision is launch configuration; HTTP model alias alone does not attest weights. Preserve server launch manifest.",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    start = time.monotonic()
    try:
        with (out / "console.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "orchestrator.cli", str(config_path), "--env", "research", "--live", "--no-color"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600,
            )
        manifest["exit_code"] = result.returncode
        manifest["status"] = "complete" if result.returncode == 0 else "incomplete"
        if result.returncode:
            raise RuntimeError("Pilot failed; inspect console.log and receipts")
        summary = summarize(out / "simulation.db")
        summary["wall_seconds"] = time.monotonic() - start
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2))
    except BaseException:
        manifest["status"] = "incomplete"
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - start
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

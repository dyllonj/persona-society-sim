"""Isolate all-clues task competence with one local model call per case."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.vllm_backend import VLLMBackend
from env.distributed_clues import DistributedClues
from evaluation.personality import strict_json
from scripts.run_distributed_clues import policy_prompt


def intersection_prompt(observation):
    # Render only available evidence. Never compute or insert the solution.
    lists = "\n".join(f"{c['id']} ({c['subject']}): {', '.join(c['allowed_plans'])}"
                      for c in observation["known_evidence"])
    return (
        "Choose a meeting plan that is allowed by EVERY constraint below. "
        "The answer must occur in ALL THREE lists, not merely in one or two. "
        "Find their set intersection and check your selected plan against each list. "
        "Exactly one plan is common to all three lists. All evidence is available here; no other agent is needed.\n"
        + lists
        + '\nReturn ONLY JSON with this structure: {"action":"submit_report","params":{"plan_id":"plan ID","evidence_ids":["all three evidence IDs"]}}. '
        "Use the actual IDs above."
    )


def run_case(backend, seed, replicate, mode):
    env = DistributedClues(seed, "all_clues_single", rounds=1)
    # Fixture advances directly to the final decision. This is not counted as
    # a model action or a full society episode.
    env.step(env.current_agent, {"action": "wait", "params": {}})
    observation = env.observe(env.current_agent)
    prompt = policy_prompt(observation) if mode == "phase-v2" else intersection_prompt(observation)
    sampling_seed = int.from_bytes(hashlib.sha256(f"single-diagnostic:{seed}:{replicate}".encode()).digest()[:4], "big")
    response = backend.generate(prompt, 256, {}, sampling_seed=sampling_seed)
    try:
        action = strict_json(response.text)
    except (ValueError, TypeError, RecursionError):
        action = None
    result = env.step(env.current_agent, action)
    return {"seed": seed, "replicate": replicate, "mode": mode, "observation": observation,
            "prompt": prompt, "sampling_seed": sampling_seed, "raw_completion": response.text,
            "action": action, "result": result, "answer": env.audit()["answer"],
            "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
            "fixture_wait_turns": 1, "model_calls": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=("phase-v2", "intersection-v1"), default=["phase-v2", "intersection-v1"])
    parser.add_argument("--purpose", choices=("development", "fresh-verification"), default="development")
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 100 or not 1 <= args.replicates <= 5:
        parser.error("Use 1..100 seeds and 1..5 replicates")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate seeds or modes")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend = VLLMBackend("local-qwen3-8b", model_revision="545dc4251c05440727734bcd94334791f6ab0192",
                          receipt_dir=out / "receipts", temperature=0.2, top_p=0.9)
    files = [Path(__file__), ROOT / "env/distributed_clues.py", ROOT / "agents/vllm_backend.py",
             ROOT / "evaluation/personality.py", ROOT / "scripts/run_distributed_clues.py"]
    manifest = {"status": "running", "real_inference": True, "kind": "single-agent-final-decision-diagnostic",
                "config": {**vars(args), "output": str(out)}, "model_revision": backend.model_revision,
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "temperature": 0.2, "top_p": 0.9, "max_new_tokens": 256, "steering": False,
                "gate": "Engineering target: >=95% valid correct supported first submissions on fresh instances. Not a mastery or generalization claim."}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    start, rows = time.monotonic(), []
    try:
        with (out / "traces.jsonl").open("w") as trace:
            for seed in args.seeds:
                for replicate in range(args.replicates):
                    for mode in args.modes:
                        row = run_case(backend, seed, replicate, mode)
                        rows.append(row)
                        trace.write(json.dumps(row) + "\n")
                        trace.flush()
                        print(json.dumps({"seed": seed, "replicate": replicate, "mode": mode, "result": row["result"]}), flush=True)
        groups = defaultdict(list)
        for row in rows:
            groups[row["mode"]].append(row)
        summary = {mode: {"cases": len(group), "unique_instances": len({r["seed"] for r in group}),
                          "valid": sum(r["result"]["valid"] for r in group),
                          "supported_correct": sum(r["result"].get("supported_success", False) for r in group),
                          "accuracy": sum(r["result"].get("supported_success", False) for r in group) / len(group),
                          "tokens_in": sum(r["tokens_in"] for r in group), "tokens_out": sum(r["tokens_out"] for r in group)}
                   for mode, group in groups.items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "incomplete"
        raise
    finally:
        manifest["completed_cases"] = len(rows)
        manifest["wall_seconds"] = time.monotonic() - start
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

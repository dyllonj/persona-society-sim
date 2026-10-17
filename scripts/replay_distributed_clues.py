"""Verify a completed local batch by replaying actions without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.distributed_clues import DistributedClues
from evaluation.personality import strict_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_episode(path, *, seed, condition, rounds, layout="together"):
    from scripts.run_distributed_clues import policy_prompt
    env = DistributedClues(seed, condition, rounds, layout=layout)
    require(json.loads((path / "evaluator-only.json").read_text()) == env.audit(), "Evaluator state mismatch")
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
    for row in rows:
        require(not env.done, "Extra turn after episode completion")
        require(row["agent"] == env.current_agent, "Actor mismatch")
        require(row["observation"] == env.observe(env.current_agent), "Observation mismatch or private-information leak")
        require(row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest(), "Prompt hash mismatch")
        require(row["prompt"] == policy_prompt(row["observation"], summary["prompt_version"]), "Full policy prompt mismatch")
        try:
            parsed = strict_json(row["raw_completion"])
        except (ValueError, TypeError, RecursionError):
            parsed = None
        require(parsed == row["action"], "Model output differs from executed action")
        env.step(env.current_agent, parsed)
        require(all(row[k] == v for k, v in env.events[-1].items()), "Replayed event mismatch")
    require(env.done, "Episode is incomplete")
    summary = json.loads((path / "summary.json").read_text())
    require(all(summary[k] == v for k, v in env.summary().items()), "Replayed score mismatch")
    require(summary["submissions"] == env.submissions, "Submission mismatch")
    for key in ("tokens_in", "tokens_out"):
        require(summary[key] == sum(row[key] for row in rows), "Token accounting mismatch")
    return len(rows)


def verify_diagnostic(path):
    from scripts.run_distributed_clues import policy_prompt
    from scripts.run_single_agent_diagnostic import intersection_prompt
    manifest = json.loads((path / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Diagnostic is incomplete")
    cfg = manifest["config"]
    expected_order = [(s, r, m) for s in cfg["seeds"] for r in range(cfg["replicates"]) for m in cfg["modes"]]
    rows = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
    require([(r["seed"], r["replicate"], r["mode"]) for r in rows] == expected_order, "Diagnostic case inventory mismatch")
    require(manifest["completed_cases"] == len(rows), "Diagnostic count mismatch")
    require(len(list((path / "receipts").glob("*.json"))) == len(rows), "Diagnostic receipt count mismatch")
    counts = {}
    for row in rows:
        env = DistributedClues(row["seed"], "all_clues_single", rounds=1)
        env.step(env.current_agent, {"action": "wait", "params": {}})
        obs = env.observe(env.current_agent)
        require(row["observation"] == obs, "Diagnostic observation mismatch")
        require(row["mode"] in ("phase-v2", "intersection-v1"), "Unsupported prompt mode")
        prompt = policy_prompt(obs) if row["mode"] == "phase-v2" else intersection_prompt(obs)
        require(row["prompt"] == prompt, "Diagnostic full prompt mismatch")
        seed = int.from_bytes(hashlib.sha256(f"single-diagnostic:{row['seed']}:{row['replicate']}".encode()).digest()[:4], "big")
        require(seed == row["sampling_seed"], "Diagnostic sampling seed mismatch")
        payload = {"model": "local-qwen3-8b", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": manifest["max_new_tokens"], "temperature": manifest["temperature"],
                   "top_p": manifest["top_p"], "chat_template_kwargs": {"enable_thinking": False}, "seed": seed}
        identity = {"backend": "local-vllm-v1", "endpoint": "http://127.0.0.1:8000/v1",
                    "model_revision": manifest["model_revision"], "request": payload}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt = json.loads((path / "receipts" / f"{digest}.json").read_text())
        require(receipt["identity"] == identity and receipt["request_sha256"] == digest, "Diagnostic request receipt mismatch")
        response = receipt["response"]
        require(response["choices"][0]["finish_reason"] == "stop" and response["model"] == "local-qwen3-8b", "Incomplete or mismatched diagnostic response")
        require(response["choices"][0]["message"]["content"] == row["raw_completion"], "Diagnostic response mismatch")
        require(response["usage"]["prompt_tokens"] == row["tokens_in"] and response["usage"]["completion_tokens"] == row["tokens_out"], "Diagnostic token mismatch")
        try:
            action = strict_json(row["raw_completion"])
        except (ValueError, TypeError, RecursionError):
            action = None
        require(action == row["action"], "Diagnostic action mismatch")
        result = env.step(env.current_agent, action)
        require(result == row["result"] and env.audit()["answer"] == row["answer"], "Diagnostic score mismatch")
        group = counts.setdefault(row["mode"], {"cases": 0, "unique_instances": len(cfg["seeds"]), "valid": 0, "supported_correct": 0, "tokens_in": 0, "tokens_out": 0})
        group["cases"] += 1
        group["valid"] += result["valid"]
        group["supported_correct"] += result.get("supported_success", False)
        group["tokens_in"] += row["tokens_in"]
        group["tokens_out"] += row["tokens_out"]
    for group in counts.values():
        group["accuracy"] = group["supported_correct"] / group["cases"]
    require(counts == json.loads((path / "summary.json").read_text()), "Diagnostic summary mismatch")
    return {"verified_model_decisions_and_receipts": len(rows), "unique_task_instances": len(cfg["seeds"]),
            "method": "Full prompt reconstruction, receipt reconciliation, deterministic regrading; no model calls"}


def verify_batch(path):
    manifest = json.loads((path / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Batch is not complete")
    config = manifest["config"]
    expected = len(config["seeds"]) * config["replicates"] * len(config["conditions"])
    require(manifest["completed_episodes"] == expected, "Episode count mismatch")
    receipts = {}
    for file in (path / "receipts").glob("*.json"):
        receipt = json.loads(file.read_text())
        identity = receipt["identity"]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        require(digest == receipt["request_sha256"] == file.stem, "Receipt identity mismatch")
        req = identity["request"]
        require(identity["model_revision"] == manifest["model_revision"], "Receipt model revision mismatch")
        require(identity["backend"] == "local-vllm-v1" and identity["endpoint"] == "http://127.0.0.1:8000/v1", "Receipt backend mismatch")
        expected_request = {"model": "local-qwen3-8b", "messages": [{"role": "user", "content": req["messages"][0]["content"]}],
                            "max_tokens": manifest["max_new_tokens"], "temperature": manifest["temperature"],
                            "top_p": manifest["top_p"], "chat_template_kwargs": {"enable_thinking": False}, "seed": req["seed"]}
        require(req == expected_request, "Receipt generation settings mismatch")
        key = (req["messages"][0]["content"], req["seed"])
        require(key not in receipts, "Duplicate request identity")
        receipts[key] = receipt
    count = 0
    used_receipts = set()
    rows = []
    for seed in config["seeds"]:
        for replicate in range(config["replicates"]):
            for condition in config["conditions"]:
                episode = path / f"seed-{seed}-rep-{replicate}-{condition}"
                count += verify_episode(episode, seed=seed, condition=condition, rounds=config["rounds"],
                                        layout=config.get("layout", "together"))
                rows.append(json.loads((episode / "summary.json").read_text()))
                require(rows[-1]["prompt_version"] == manifest["prompt_version"], "Episode prompt version mismatch")
                for line in (episode / "trace.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    seed_text = f"{seed}:{replicate}:{row['round']}:{row['agent']}:{row['phase']}"
                    expected_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "big")
                    require(row["sampling_seed"] == expected_seed, "Episode sampling seed mismatch")
                    key = (row["prompt"], row["sampling_seed"])
                    receipt = receipts.get(key)
                    used_receipts.add(key)
                    require(receipt is not None, "Missing receipt for executed action")
                    response = receipt["response"]
                    require(response["model"] == "local-qwen3-8b" and len(response["choices"]) == 1 and response["choices"][0]["finish_reason"] == "stop", "Incomplete or mismatched response")
                    require(response["choices"][0]["message"]["content"] == row["raw_completion"], "Response differs from trace")
                    require(response["usage"]["prompt_tokens"] == row["tokens_in"], "Receipt input token mismatch")
                    require(response["usage"]["completion_tokens"] == row["tokens_out"], "Receipt output token mismatch")
    require(set(receipts) == used_receipts, "Unaccounted-for receipts")
    recorded = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    require(recorded == rows, "Batch episode ledger mismatch")
    from scripts.run_distributed_clues import aggregate
    require(json.loads((path / "comparison.json").read_text()) == aggregate(rows), "Aggregate score mismatch")
    return {"verified_episodes": expected, "verified_actions_and_receipts": count,
            "unique_inference_receipts": len(receipts), "reused_decisions": count - len(receipts),
            "unique_inference_tokens": {k: sum(r["response"]["usage"][k] for r in receipts.values()) for k in ("prompt_tokens", "completion_tokens")},
            "method": "Deterministic replay and receipt reconciliation; no inference calls"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    verify = verify_diagnostic if args.diagnostic else verify_batch
    print(json.dumps(verify(args.batch.resolve()), indent=2))

"""Analyze one or more completed distributed-clue batches as a single paired study.

Episodes on the same environment seed share a task instance, and residents within
an episode share a conversation. The task instance (seed) is therefore the
resampling unit for confidence intervals; replicates and residents are averaged
within it first. Only batches whose manifest status is ``complete`` are accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261017


def load_batches(paths):
    rows, identity = [], None
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"{path} is not a complete batch")
        batch_identity = (manifest["task_version"], manifest["prompt_version"], manifest["model_revision"])
        if identity is None:
            identity = batch_identity
        elif batch_identity != identity:
            raise ValueError(f"{path} mixes task, prompt or model versions: {batch_identity} != {identity}")
        for line in (path / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["batch"] = path.name
            rows.append(row)
    keys = [(r["seed"], r["replicate"], r["condition"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate seed/replicate/condition across batches")
    return rows, identity


def classify_submissions(row):
    """Count each resident's outcome once, separating delivery from synthesis failures."""
    counts = defaultdict(int)
    coverage = row["evidence_coverage"]
    for resident in coverage:
        sub = row["submissions"].get(resident)
        if not sub or not sub.get("valid"):
            counts["invalid_or_missing"] += 1
        elif sub["answer_correct"] and sub["evidence_complete"]:
            counts["correct_supported"] += 1
        elif sub["answer_correct"] and coverage[resident] < 1.0:
            counts["correct_without_all_clues"] += 1
        elif sub["answer_correct"]:
            counts["correct_but_evidence_not_cited"] += 1
        elif coverage[resident] < 1.0:
            counts["wrong_missing_clues"] += 1
        else:
            counts["wrong_despite_all_clues"] += 1
    return counts


def plurality_vote(row):
    """Fixed aggregator over independent submissions: no extra inference.

    Returns (strict, expected): strict credits only a unique top plan that is
    correct; expected credits a uniformly random tie-break among top plans.
    """
    votes = defaultdict(int)
    correct_plans = set()
    for sub in row["submissions"].values():
        if sub.get("valid"):
            votes[sub["plan_id"]] += 1
            if sub["answer_correct"]:
                correct_plans.add(sub["plan_id"])
    if not votes:
        return 0.0, 0.0
    top = max(votes.values())
    leaders = [plan for plan, count in votes.items() if count == top]
    hits = sum(plan in correct_plans for plan in leaders)
    return float(len(leaders) == 1 and hits == 1), hits / len(leaders)


def guess_diagnostics(paths):
    """How residents choose under missing information, using evaluator-only files.

    For each valid submission from a resident holding fewer than all clues, record
    whether the plan satisfies every clue it holds and its position in the
    plan-ordered candidate list. Uniform choice among consistent candidates is
    the chance reference; it is not a policy the model is asked to follow.
    """
    positions, consistent, total, chance = defaultdict(int), 0, 0, []
    for path in paths:
        for episode in sorted(path.glob("seed-*")):
            summary = json.loads((episode / "summary.json").read_text())
            if summary["condition"] == "all_clues_single":
                continue
            audit = json.loads((episode / "evaluator-only.json").read_text())
            trace = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()]
            known = {row["agent"]: set(row["known_before"]) for row in trace if row["phase"] == "submit"}
            for agent, sub in summary["submissions"].items():
                held = known.get(agent, set())
                if not sub.get("valid") or len(held) == len(audit["clues"]):
                    continue
                candidates = sorted(set.intersection(*(set(audit["clues"][e]["allowed_plans"]) for e in held)),
                                    key=lambda p: int(p.split("_")[1]))
                total += 1
                chance.append(1 / len(candidates))
                if sub["plan_id"] in candidates:
                    consistent += 1
                    positions[candidates.index(sub["plan_id"])] += 1
    if not total:
        return None
    return {"partial_information_submissions": total,
            "consistent_with_held_clues": consistent / total,
            "candidate_position_counts": dict(sorted(positions.items())),
            "uniform_guess_expected_accuracy": statistics.fmean(chance)}


def episode_accuracy(row):
    return row["correct_answers"] / row["agents"]


def per_seed_means(rows, value):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(value(row))
    return {seed: statistics.fmean(values) for seed, values in by_seed.items()}


def bootstrap_ci(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def variance_components(rows):
    """One-way random-effects estimate: between-instance vs within-instance (replicate) variance."""
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(episode_accuracy(row))
    groups = [v for v in by_seed.values() if len(v) >= 2]
    if len(groups) < 2 or len({len(v) for v in groups}) != 1:
        return None
    k, n = len(groups), len(groups[0])
    grand = statistics.fmean(x for g in groups for x in g)
    ms_between = n * sum((statistics.fmean(g) - grand) ** 2 for g in groups) / (k - 1)
    ms_within = sum((x - statistics.fmean(g)) ** 2 for g in groups for x in g) / (k * (n - 1))
    between = max(0.0, (ms_between - ms_within) / n)
    total = between + ms_within
    return {"instances": k, "replicates_per_instance": n,
            "between_instance_variance": between, "within_instance_variance": ms_within,
            "instance_share": between / total if total else None}


def summarize(rows):
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    conditions = {}
    for condition, items in sorted(by_condition.items()):
        seed_acc = list(per_seed_means(items, episode_accuracy).values())
        failures = defaultdict(int)
        for row in items:
            for key, count in classify_submissions(row).items():
                failures[key] += count
        conditions[condition] = {
            "episodes": len(items),
            "instances": len({r["seed"] for r in items}),
            "submitters": sum(r["agents"] for r in items),
            "answer_accuracy": sum(r["correct_answers"] for r in items) / sum(r["agents"] for r in items),
            "answer_accuracy_ci95_by_instance": bootstrap_ci(seed_acc),
            "supported_success": sum(r["supported_submissions"] for r in items) / sum(r["agents"] for r in items),
            "team_success_rate": statistics.fmean(bool(r["team_success"]) for r in items),
            "invalid_action_rate": sum(r["invalid_actions"] for r in items) / sum(r["turns_executed"] for r in items),
            "mean_evidence_coverage": statistics.fmean(
                statistics.fmean(r["evidence_coverage"].values()) for r in items),
            "submission_outcomes": dict(sorted(failures.items())),
            "tokens_in": sum(r["tokens_in"] for r in items),
            "tokens_out": sum(r["tokens_out"] for r in items),
            "wall_seconds": sum(r["wall_seconds"] for r in items),
            "variance_components": variance_components(items),
        }
    paired = {}
    on = per_seed_means(by_condition.get("communication", []), episode_accuracy)
    off = per_seed_means(by_condition.get("no_communication", []), episode_accuracy)
    shared = sorted(set(on) & set(off))
    if shared:
        diffs = [on[s] - off[s] for s in shared]
        paired = {"comparison": "communication minus no_communication answer accuracy",
                  "instances": len(shared), "mean_difference": statistics.fmean(diffs),
                  "ci95_instance_bootstrap": bootstrap_ci(diffs),
                  "instances_favoring_communication": sum(d > 0 for d in diffs),
                  "instances_tied": sum(d == 0 for d in diffs)}
    independent = by_condition.get("no_communication", [])
    if independent:
        votes = [plurality_vote(r) for r in independent]
        strict_by_seed = per_seed_means(independent, lambda r: plurality_vote(r)[0])
        paired["independent_vote_baseline"] = {
            "source": "plurality over no_communication submissions; same inference calls, no new samples",
            "strict_accuracy": statistics.fmean(v[0] for v in votes),
            "tie_break_expected_accuracy": statistics.fmean(v[1] for v in votes),
            "communication_minus_strict_vote": (
                statistics.fmean(on[s] - strict_by_seed[s] for s in shared) if shared else None),
            "ci95_instance_bootstrap": bootstrap_ci([on[s] - strict_by_seed[s] for s in shared]) if shared else None,
        }
    return {"conditions": conditions, "paired_communication_effect": paired,
            "resampling_unit": "task instance (environment seed)",
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "method": "percentile"},
            "note": "One synthetic task family and one local model. all_clues_single uses fewer calls and is a "
                    "solvability reference, not a matched-compute baseline. Supported success is unattainable "
                    "without communication by construction; compare answer accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, identity = load_batches(args.batches)
    result = {"batches": [p.name for p in args.batches], "task_version": identity[0],
              "prompt_version": identity[1], "model_revision": identity[2], **summarize(rows),
              "partial_information_guessing": guess_diagnostics(args.batches)}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

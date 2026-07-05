"""Run neutral, placebo, and targeted steering arms with matched probes/metrics."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from uuid import uuid4

import numpy as np
import torch

from orchestrator.cli import (
    TRAIT_KEYS,
    build_agents,
    build_language_backend,
    build_meta_orchestrator,
    build_objective_manager,
    build_probe_manager,
    load_config,
    load_trait_vectors,
    shuffle_trait_vectors,
    _load_metadata_file,
    _steering_coefficients,
)
from orchestrator.console_logger import ConsoleLogger
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Scheduler
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.logs import ActionLog, MetricsSnapshot
from storage.log_sink import LogSink
from env.world import World


class CollectingLogSink(LogSink):
    """Log sink that preserves metrics and actions for in-memory analysis."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics_snapshots: List[MetricsSnapshot] = []
        self.actions: List[ActionLog] = []

    def log_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:  # type: ignore[override]
        self.metrics_snapshots.append(snapshot)
        super().log_metrics_snapshot(snapshot)

    def log_action(self, log: ActionLog) -> None:  # type: ignore[override]
        self.actions.append(log)
        super().log_action(log)


def _prepare_runner(
    base_config: Dict[str, Any],
    *,
    config_path: Path,
    env_choice: str,
    difficulty: int,
    steering_mode: str,
    vector_dir: Path,
    run_suffix: str,
    placebo_seed: int,
) -> Tuple[SimulationRunner, CollectingLogSink, int, Mapping[str, str]]:
    config = copy.deepcopy(base_config)
    run_root = config.get("run_id") or f"run-{uuid4().hex[:6]}"
    run_id = f"{run_root}-{run_suffix}"
    seed = config.get("seed", 7)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world = World()
    world.configure_environment(env_choice, difficulty)
    scheduler = Scheduler(world, seed=seed)

    steering_cfg = config.get("steering", {})
    steering_enabled = steering_mode != "disabled" and steering_cfg.get("enabled", True)
    steering_base = _steering_coefficients(steering_cfg) if steering_enabled else {}
    metadata_files = steering_cfg.get("metadata_files") or {}
    vector_metadata = _load_metadata_file(metadata_files.get("vectors"), config_dir=config_path.parent)
    trait_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    vector_norms: Dict[str, Dict[int, float]] = {}
    shuffle_mapping: Mapping[str, str] = {}
    if steering_enabled:
        trait_vectors, vector_norms = load_trait_vectors(
            list(steering_base.keys() or TRAIT_KEYS),
            vector_dir,
            vector_metadata=vector_metadata,
        )
        if steering_mode == "placebo":
            rng = random.Random(placebo_seed)
            trait_vectors, vector_norms, shuffle_mapping = shuffle_trait_vectors(
                trait_vectors, vector_norms, rng
            )
    safety_cfg = config.get("safety", {})
    safety = SafetyGovernor(
        SafetyConfig(
            alpha_clip=safety_cfg.get("alpha_clip", 1.0),
            toxicity_threshold=safety_cfg.get("toxicity_threshold", 0.4),
            governor_backoff=safety_cfg.get("governor_backoff", 0.2),
            global_alpha_strength=steering_cfg.get("strength", 1.0) if steering_enabled else 0.0,
        )
    )

    backend = build_language_backend(
        config,
        trait_vectors,
        vector_norms,
        mock=config.get("mock_model", False),
        suppress_alphas=not steering_enabled,
    )

    agents = build_agents(
        run_id,
        config,
        world,
        backend,
        safety,
        env_choice,
        config_dir=config_path.parent,
        suppress_alphas=not steering_enabled,
    )

    logging_cfg = config.get("logging", {})
    log_sink: CollectingLogSink = CollectingLogSink(
        run_id,
        logging_cfg.get("db_url"),
        logging_cfg.get("parquet_dir"),
    )

    inference_cfg = config.get("inference", {})
    objective_manager = build_objective_manager(config, env_choice, difficulty)
    probe_manager = build_probe_manager(config)
    console_logger = ConsoleLogger(enabled=False, use_colors=False, truncate=True)
    meta_orchestrator = build_meta_orchestrator(
        config, env_choice, config_dir=config_path.parent
    )

    runner = SimulationRunner(
        run_id=run_id,
        world=world,
        scheduler=scheduler,
        agents=agents,
        log_sink=log_sink,
        temperature=inference_cfg.get("temperature", 0.7),
        top_p=inference_cfg.get("top_p", 0.9),
        console_logger=console_logger,
        objective_manager=objective_manager,
        probe_manager=probe_manager,
        event_bridge=None,
        meta_orchestrator=meta_orchestrator,
    )
    steps = int(config.get("steps", 200))
    return runner, log_sink, steps, shuffle_mapping


def _aggregate_macro_metrics(metrics: Iterable[MetricsSnapshot]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[MetricsSnapshot]] = {}
    for snapshot in metrics:
        key = snapshot.trait_key or "global"
        buckets.setdefault(key, []).append(snapshot)

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    aggregates: Dict[str, Dict[str, float]] = {}
    for key, snaps in buckets.items():
        aggregates[key] = {
            "cooperation_rate": _avg([s.cooperation_rate for s in snaps]),
            "gini_wealth": _avg([s.gini_wealth for s in snaps]),
            "polarization_modularity": _avg([s.polarization_modularity for s in snaps]),
            "prompt_duplication_rate": _avg([s.prompt_duplication_rate for s in snaps]),
            "plan_reuse_rate": _avg([s.plan_reuse_rate for s in snaps]),
            "conflicts": _avg([float(s.conflicts) for s in snaps]),
            "rule_enforcement_cost": _avg([s.rule_enforcement_cost for s in snaps]),
        }
    return aggregates


def _summarize(log_sink: CollectingLogSink) -> Dict[str, Any]:
    global_metrics = [m for m in log_sink.metrics_snapshots if m.trait_key is None]
    collab_ratio = global_metrics[-1].cooperation_rate if global_metrics else 0.0
    prompt_dup_rate = max((m.prompt_duplication_rate for m in global_metrics), default=0.0)
    task_completions = sum(
        1
        for action in log_sink.actions
        if action.action_type == "submit_report" and action.outcome == "success"
    )
    return {
        "collab_ratio": round(collab_ratio, 3),
        "prompt_duplication": round(prompt_dup_rate, 3),
        "task_completions": task_completions,
        "macro_metrics": _aggregate_macro_metrics(log_sink.metrics_snapshots),
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["mode", "collab_ratio", "prompt_duplication", "task_completions"]
    widths = [max(len(h), max(len(str(row.get(h, ""))) for row in rows)) for h in headers]

    def _fmt(row: Dict[str, Any]) -> str:
        return " | ".join(str(row.get(h, "")).ljust(widths[idx]) for idx, h in enumerate(headers))

    print(" | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(_fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run neutral/placebo/targeted steering comparisons with matched probes and metrics"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--env", choices=["research", "policy", "nav"], default="research")
    parser.add_argument("--difficulty", type=int, default=3, help="Environment difficulty setting")
    parser.add_argument("--steps", type=int, help="Override simulation steps for all runs")
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("data/vectors"),
        help="Directory containing steering vectors",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["targeted", "placebo", "disabled"],
        default=["targeted", "placebo", "disabled"],
        help="Which steering presets to run (default: run all three)",
    )
    parser.add_argument(
        "--placebo-seed",
        type=int,
        help="Seed used to shuffle trait/vector assignments for placebo runs",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Tolerance for treating placebo and neutral macro metrics as matched",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON comparison summary",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.steps is not None:
        base_config["steps"] = int(args.steps)

    arms = args.arms or ["targeted", "placebo", "disabled"]
    results: Dict[str, Dict[str, Any]] = {}
    shuffle_maps: Dict[str, Mapping[str, str]] = {}
    placebo_seed = args.placebo_seed if args.placebo_seed is not None else base_config.get("seed", 7)
    for arm in arms:
        runner, sink, steps, shuffle_mapping = _prepare_runner(
            base_config,
            config_path=args.config,
            env_choice=args.env,
            difficulty=args.difficulty,
            steering_mode=arm,
            vector_dir=args.vector_dir,
            run_suffix=arm,
            placebo_seed=placebo_seed,
        )
        runner.run(
            base_config.get("steps", steps),
            max_events_per_tick=base_config.get("max_events_per_tick", base_config.get("max_events", 16)),
        )
        summary = {"mode": arm, **_summarize(sink)}
        if shuffle_mapping:
            summary["placebo_mapping"] = shuffle_mapping
            shuffle_maps[arm] = shuffle_mapping
        results[arm] = summary

    rows = list(results.values())
    print("Steering arm comparison (seed=%s, env=%s)" % (base_config.get("seed", 7), args.env))
    _print_table(rows)

    neutral = results.get("disabled") or {}
    placebo = results.get("placebo") or {}
    targeted = results.get("targeted") or {}

    def _metric_gaps(source: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
        keys = ["collab_ratio", "prompt_duplication", "task_completions"]
        return {k: round(float(source.get(k, 0.0)) - float(baseline.get(k, 0.0)), 3) for k in keys}

    placebo_deltas = _metric_gaps(placebo, neutral) if neutral else {}
    placebo_matches = all(abs(v) <= args.tolerance for v in placebo_deltas.values()) if placebo_deltas else False

    trait_deltas: Dict[str, float] = {}
    targeted_macros = targeted.get("macro_metrics", {}) if targeted else {}
    neutral_macros = neutral.get("macro_metrics", {}) if neutral else {}
    for cohort, values in targeted_macros.items():
        base = neutral_macros.get(cohort, {})
        trait_deltas[cohort] = round(values.get("cooperation_rate", 0.0) - base.get("cooperation_rate", 0.0), 3)

    summary_payload = {
        "env": args.env,
        "seed": base_config.get("seed", 7),
        "runs": rows,
        "placebo_vs_neutral": {
            "deltas": placebo_deltas,
            "tolerance": args.tolerance,
            "matched": placebo_matches,
        },
        "targeted_vs_neutral_trait_deltas": trait_deltas,
        "placebo_mapping": shuffle_maps,
    }

    print("\nPlacebo vs neutral deltas (<= tolerance considered match):")
    print(json.dumps(placebo_deltas, indent=2))
    print(f"Matched within tolerance: {placebo_matches}")

    print("\nTargeted vs neutral cooperation deltas by dominant trait cohort:")
    print(json.dumps(trait_deltas, indent=2))

    if args.output:
        args.output.write_text(json.dumps(summary_payload, indent=2))
        print(f"Wrote summary to {args.output}")
    else:
        print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()

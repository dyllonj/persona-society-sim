"""Run paired simulations with and without persona steering for quick baseline checks."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
import torch

from orchestrator.cli import (
    TRAIT_KEYS,
    build_agents,
    build_language_backend,
    build_objective_manager,
    build_probe_manager,
    load_config,
    load_trait_vectors,
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
    steering_enabled: bool,
    vector_dir: Path,
    run_suffix: str,
) -> Tuple[SimulationRunner, CollectingLogSink, int]:
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
    steering_base = _steering_coefficients(steering_cfg) if steering_enabled else {}
    metadata_files = steering_cfg.get("metadata_files") or {}
    vector_metadata = _load_metadata_file(metadata_files.get("vectors"), config_dir=config_path.parent)
    trait_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    vector_norms: Dict[str, Dict[int, float]] = {}
    if steering_enabled:
        trait_vectors, vector_norms = load_trait_vectors(
            list(steering_base.keys() or TRAIT_KEYS),
            vector_dir,
            vector_metadata=vector_metadata,
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
    )
    steps = int(config.get("steps", 200))
    return runner, log_sink, steps


def _summarize(log_sink: CollectingLogSink) -> Dict[str, float]:
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
    parser = argparse.ArgumentParser(description="Run steering vs no-steering baseline comparisons")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--env", choices=["research", "policy", "nav"], default="research")
    parser.add_argument("--difficulty", type=int, default=3, help="Environment difficulty setting")
    parser.add_argument("--steps", type=int, help="Override simulation steps for both runs")
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("data/vectors"),
        help="Directory containing steering vectors",
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

    steering_runner, steering_sink, steps = _prepare_runner(
        base_config,
        config_path=args.config,
        env_choice=args.env,
        difficulty=args.difficulty,
        steering_enabled=True,
        vector_dir=args.vector_dir,
        run_suffix="steering",
    )
    steering_runner.run(base_config.get("steps", steps), max_events_per_tick=base_config.get("max_events", 16))

    no_steer_runner, no_steer_sink, steps = _prepare_runner(
        base_config,
        config_path=args.config,
        env_choice=args.env,
        difficulty=args.difficulty,
        steering_enabled=False,
        vector_dir=args.vector_dir,
        run_suffix="neutral",
    )
    no_steer_runner.run(base_config.get("steps", steps), max_events_per_tick=base_config.get("max_events", 16))

    steering_summary = {"mode": "steering", **_summarize(steering_sink)}
    neutral_summary = {"mode": "no_steering", **_summarize(no_steer_sink)}
    rows = [steering_summary, neutral_summary]

    print("Baseline comparison (seed=%s, env=%s)" % (base_config.get("seed", 7), args.env))
    _print_table(rows)

    summary_payload = {"env": args.env, "seed": base_config.get("seed", 7), "runs": rows}
    if args.output:
        args.output.write_text(json.dumps(summary_payload, indent=2))
        print(f"Wrote summary to {args.output}")
    else:
        print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()

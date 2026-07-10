"""Analyze society-study outputs with the simulation run as the sampling unit."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_society_study import DEFAULT_MATRIX, load_matrix, sha256_file


RUN_OUTCOME_FIELDS = (
    "cooperation_rate",
    "gini_wealth",
    "polarization_modularity",
    "conflicts_total",
    "rule_enforcement_cost_total",
    "prompt_duplication_rate",
    "plan_reuse_rate",
    "action_type_entropy",
    "action_success_rate",
    "task_completions",
    "research_fact_accuracy",
    "citations_total",
    "report_reward_mean",
    "tokens_total",
    "n_actions",
)

# Two-sided 95% Student-t critical values. The preregistered study has df=4;
# the rest make the report helper correct for smaller or modestly larger pilots
# without introducing another runtime dependency.
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_parquet_rows(output_dir: Path, kind: str) -> list[dict[str, Any]]:
    directory = output_dir / kind
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _validate_row_run_ids(rows: Iterable[dict[str, Any]], run_id: str, kind: str) -> None:
    wrong = sorted(
        {str(row.get("run_id")) for row in rows if row.get("run_id") != run_id}
    )
    if wrong:
        raise ValueError(f"{kind} contains rows from other runs: {wrong}")


def _global_metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("trait_key") in {None, "global"}]
    ticks = [int(row["tick"]) for row in selected]
    if len(ticks) != len(set(ticks)):
        raise ValueError("global metrics contain duplicate ticks; outputs may have been pooled")
    return sorted(selected, key=lambda row: int(row["tick"]))


def aggregate_one_run(
    *,
    run_id: str,
    arm: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Collapse all ticks/actions into exactly one observational row."""
    metrics = _read_parquet_rows(output_dir, "metrics_snapshots")
    actions = _read_parquet_rows(output_dir, "actions")
    messages = _read_parquet_rows(output_dir, "messages")
    facts = _read_parquet_rows(output_dir, "research_facts")
    citations = _read_parquet_rows(output_dir, "citations")
    grades = _read_parquet_rows(output_dir, "report_grades")
    for kind, rows in (
        ("metrics", metrics),
        ("actions", actions),
        ("messages", messages),
        ("research_facts", facts),
        ("citations", citations),
        ("report_grades", grades),
    ):
        _validate_row_run_ids(rows, run_id, kind)

    global_metrics = _global_metric_rows(metrics)
    if not global_metrics:
        raise ValueError(f"run {run_id} has no global metric snapshots")
    successes = sum(row.get("outcome") == "success" for row in actions)
    task_completions = sum(
        row.get("action_type") == "submit_report" and row.get("outcome") == "success"
        for row in actions
    )
    correct_facts = sum(bool(row.get("correct")) for row in facts)
    fact_accuracy = correct_facts / len(facts) if facts else None
    result = {
        "analysis_unit": "simulation_run",
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "ticks_observed": len(global_metrics),
        "cooperation_rate": _mean(float(row["cooperation_rate"]) for row in global_metrics),
        "gini_wealth": _mean(float(row["gini_wealth"]) for row in global_metrics),
        "polarization_modularity": _mean(
            float(row["polarization_modularity"]) for row in global_metrics
        ),
        "conflicts_total": sum(float(row["conflicts"]) for row in global_metrics),
        "rule_enforcement_cost_total": sum(
            float(row["rule_enforcement_cost"]) for row in global_metrics
        ),
        "prompt_duplication_rate": _mean(
            float(row.get("prompt_duplication_rate") or 0.0) for row in global_metrics
        ),
        "plan_reuse_rate": _mean(
            float(row.get("plan_reuse_rate") or 0.0) for row in global_metrics
        ),
        "action_type_entropy": _mean(
            float(row.get("action_type_entropy") or 0.0) for row in global_metrics
        ),
        "action_success_rate": successes / len(actions) if actions else None,
        "task_completions": float(task_completions),
        "research_fact_accuracy": fact_accuracy,
        "citations_total": float(len(citations)),
        "report_reward_mean": _mean(float(row["reward_points"]) for row in grades),
        "tokens_total": float(
            sum(int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0) for row in messages)
        ),
        "n_actions": float(len(actions)),
    }
    return result


def _validate_run_artifact(
    manifest_path: Path,
    *,
    matrix_sha256: str,
    require_completion: bool,
) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent.resolve()
    if manifest.get("analysis_unit") != "simulation_run":
        raise ValueError(f"run manifest has invalid analysis unit: {manifest_path}")
    if manifest.get("matrix_sha256") != matrix_sha256:
        raise ValueError(f"run manifest belongs to another matrix: {manifest_path}")
    config_path = run_dir / "config.yaml"
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise ValueError(f"run config hash mismatch: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid run config: {config_path}")
    completion_path = run_dir / "completed.json"
    if require_completion and not completion_path.is_file():
        raise ValueError(f"run is missing completion marker: {run_dir}")
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("run_id") != manifest["run_id"]:
            raise ValueError(f"completion/run manifest mismatch: {run_dir}")
        if completion.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError(f"completion/config hash mismatch: {run_dir}")
        if completion.get("run_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"completion/manifest hash mismatch: {run_dir}")
    output_dir = Path(config["logging"]["parquet_dir"]).resolve()
    if not output_dir.is_relative_to(run_dir):
        raise ValueError(f"run output path escapes its immutable run directory: {output_dir}")
    return manifest, output_dir


def load_run_outcomes(
    *,
    output_root: Path,
    matrix_path: Path,
    matrix: dict[str, Any],
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    matrix_sha256 = sha256_file(matrix_path)
    manifest_paths = sorted((output_root / "runs").glob("*/seed-*/run-manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"no run manifests found under {output_root}")
    outcomes: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    used_manifests: list[Path] = []
    errors: list[str] = []
    for path in manifest_paths:
        try:
            manifest, output_dir = _validate_run_artifact(
                path,
                matrix_sha256=matrix_sha256,
                require_completion=not allow_incomplete,
            )
            key = (str(manifest["arm"]), int(manifest["seed"]))
            if key in seen:
                raise ValueError(f"duplicate run for arm/seed {key}")
            outcome = aggregate_one_run(
                run_id=str(manifest["run_id"]),
                arm=key[0],
                seed=key[1],
                output_dir=output_dir,
            )
            if outcome["ticks_observed"] != int(manifest["steps"]):
                raise ValueError(
                    f"run {manifest['run_id']} has {outcome['ticks_observed']} global ticks; "
                    f"expected {manifest['steps']}"
                )
            outcomes.append(outcome)
            seen.add(key)
            used_manifests.append(path)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            if not allow_incomplete:
                raise
            errors.append(f"{path}: {exc}")

    if not allow_incomplete:
        expected = {
            (arm, int(seed)) for arm in matrix["arms"] for seed in matrix["seeds"]
        }
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise ValueError(f"study run matrix mismatch; missing={missing}, extra={extra}")
    if not outcomes:
        raise ValueError(f"no analyzable runs found; ignored errors: {errors}")
    return sorted(outcomes, key=lambda row: (row["arm"], row["seed"])), used_manifests


def _sample_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    n = len(values)
    if not values:
        return {"n_runs": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if n < 2:
        return {"n_runs": n, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = statistics.stdev(values)
    critical = T975.get(n - 1, 1.96)
    half_width = critical * sd / math.sqrt(n)
    return {
        "n_runs": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_arm_outcomes(
    per_run: Sequence[dict[str, Any]], outcomes: Sequence[str]
) -> list[dict[str, Any]]:
    """Compute intervals across run-level scalars, never across action rows."""
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        if row.get("analysis_unit") != "simulation_run":
            raise ValueError("arm summaries require one row per simulation run")
        by_arm[str(row["arm"])].append(row)
    summaries: list[dict[str, Any]] = []
    for arm, rows in sorted(by_arm.items()):
        for outcome in outcomes:
            values = [float(row[outcome]) for row in rows if row.get(outcome) is not None]
            summaries.append(
                {
                    "analysis_unit": "simulation_run",
                    "arm": arm,
                    "outcome": outcome,
                    **_sample_summary(values),
                }
            )
    return summaries


def paired_seed_contrasts(
    per_run: Sequence[dict[str, Any]],
    outcomes: Sequence[str],
    contrasts: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    index = {(str(row["arm"]), int(row["seed"])): row for row in per_run}
    rows: list[dict[str, Any]] = []
    all_seeds = sorted({int(row["seed"]) for row in per_run})
    for pair in contrasts:
        if len(pair) != 2:
            raise ValueError(f"paired contrast must contain two arms: {pair}")
        treatment, reference = str(pair[0]), str(pair[1])
        for outcome in outcomes:
            differences: list[float] = []
            paired_seeds: list[int] = []
            for seed in all_seeds:
                left = index.get((treatment, seed))
                right = index.get((reference, seed))
                if not left or not right:
                    continue
                if left.get(outcome) is None or right.get(outcome) is None:
                    continue
                differences.append(float(left[outcome]) - float(right[outcome]))
                paired_seeds.append(seed)
            rows.append(
                {
                    "analysis_unit": "paired_simulation_seed",
                    "contrast": f"{treatment}-minus-{reference}",
                    "treatment_arm": treatment,
                    "reference_arm": reference,
                    "outcome": outcome,
                    "paired_seeds": paired_seeds,
                    **_sample_summary(differences),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def analyze(
    *,
    output_root: Path,
    matrix_path: Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    per_run, manifests = load_run_outcomes(
        output_root=output_root,
        matrix_path=matrix_path,
        matrix=matrix,
        allow_incomplete=allow_incomplete,
    )
    outcomes = list(matrix["analysis"]["outcomes"])
    unknown = sorted(set(outcomes) - set(RUN_OUTCOME_FIELDS))
    if unknown:
        raise ValueError(f"matrix requests unknown outcomes: {unknown}")
    arm_summary = summarize_arm_outcomes(per_run, outcomes)
    contrasts = paired_seed_contrasts(
        per_run,
        outcomes,
        matrix["analysis"].get("paired_contrasts") or [],
    )
    analysis_dir = output_root / "analysis"
    outputs = {
        "per_run_json": analysis_dir / "per-run-outcomes.json",
        "per_run_csv": analysis_dir / "per-run-outcomes.csv",
        "arm_json": analysis_dir / "arm-summary.json",
        "arm_csv": analysis_dir / "arm-summary.csv",
        "contrast_json": analysis_dir / "paired-seed-contrasts.json",
        "contrast_csv": analysis_dir / "paired-seed-contrasts.csv",
    }
    _write_atomic(outputs["per_run_json"], _json_bytes(per_run))
    _write_csv(outputs["per_run_csv"], per_run)
    _write_atomic(outputs["arm_json"], _json_bytes(arm_summary))
    _write_csv(outputs["arm_csv"], arm_summary)
    _write_atomic(outputs["contrast_json"], _json_bytes(contrasts))
    _write_csv(outputs["contrast_csv"], contrasts)
    report_manifest = {
        "schema_version": "1.0",
        "analysis_unit": "simulation_run",
        "inference_warning": (
            "Ticks, agents, messages, and actions are aggregated within runs; "
            "uncertainty is calculated only across independent simulation runs."
        ),
        "matrix_sha256": sha256_file(matrix_path),
        "run_manifest_sha256": [sha256_file(path) for path in manifests],
        "n_runs": len(per_run),
        "n_arms": len({row["arm"] for row in per_run}),
        "outputs": {name: sha256_file(path) for name, path in outputs.items()},
    }
    _write_atomic(analysis_dir / "analysis-manifest.json", _json_bytes(report_manifest))
    return report_manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/society_study")
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="analyze only valid complete-looking runs instead of requiring the full matrix",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = analyze(
        output_root=args.output_root.resolve(),
        matrix_path=args.matrix.resolve(),
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

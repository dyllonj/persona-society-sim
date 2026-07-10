"""Materialize and sequentially execute the preregistered society-study matrix.

This command never provisions compute. Real or mock simulation execution is
impossible unless ``--execute`` is supplied explicitly.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.cli import deranged_trait_mapping, load_config


DEFAULT_MATRIX = Path("experiments/society_study/matrix.yaml")


@dataclass(frozen=True)
class RunSpec:
    study_id: str
    arm: str
    seed: int
    steering_mode: str
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    environment: str
    difficulty: int
    max_events_per_tick: int
    vector_dir: Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_repo_path(value: Any, repo_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (repo_root / path).resolve())


def _normalize_config_paths(config: dict[str, Any], repo_root: Path) -> None:
    steering = config.get("steering") or {}
    metadata = steering.get("metadata_files") or {}
    for key in ("personas", "vectors"):
        if key in metadata:
            metadata[key] = _resolve_repo_path(metadata[key], repo_root)
    meta = config.get("meta_orchestrator") or {}
    if "playbook_file" in meta:
        meta["playbook_file"] = _resolve_repo_path(meta["playbook_file"], repo_root)
    probes = config.get("probes") or {}
    if "definitions_path" in probes:
        probes["definitions_path"] = _resolve_repo_path(
            probes["definitions_path"], repo_root
        )


def load_matrix(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"study matrix must be a YAML mapping: {path}")
    required = {
        "schema_version",
        "study_id",
        "base_config",
        "population",
        "steps",
        "seeds",
        "arms",
        "model",
        "estimation",
        "analysis",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"study matrix is missing fields: {missing}")
    seeds = payload["seeds"]
    arms = payload["arms"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("matrix seeds must be a non-empty unique list")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("matrix arms must be a non-empty mapping")
    if payload.get("analysis", {}).get("unit") != "simulation_run":
        raise ValueError("analysis.unit must be simulation_run")
    for arm, arm_config in arms.items():
        mode = (arm_config or {}).get("steering_mode")
        if mode not in {"disabled", "targeted", "placebo"}:
            raise ValueError(f"arm {arm!r} has invalid steering_mode {mode!r}")
    return payload


def _selected_matrix(
    matrix: dict[str, Any],
    *,
    only_arms: set[str] | None,
    only_seeds: set[int] | None,
) -> tuple[list[str], list[int]]:
    all_arms = list(matrix["arms"])
    all_seeds = [int(seed) for seed in matrix["seeds"]]
    if only_arms:
        unknown = sorted(only_arms - set(all_arms))
        if unknown:
            raise ValueError(f"unknown requested arms: {unknown}")
        all_arms = [arm for arm in all_arms if arm in only_arms]
    if only_seeds:
        unknown_seeds = sorted(only_seeds - set(all_seeds))
        if unknown_seeds:
            raise ValueError(f"unknown requested seeds: {unknown_seeds}")
        all_seeds = [seed for seed in all_seeds if seed in only_seeds]
    return all_arms, all_seeds


def _placebo_mapping(active_traits: Sequence[str], seed: int) -> dict[str, str]:
    return deranged_trait_mapping(list(active_traits), random.Random(seed))


def build_run_specs(
    matrix: dict[str, Any],
    *,
    matrix_path: Path,
    output_root: Path,
    repo_root: Path = PROJECT_ROOT,
    only_arms: set[str] | None = None,
    only_seeds: set[int] | None = None,
) -> list[RunSpec]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = (repo_root / base_path).resolve()
    base_config = load_config(base_path)
    arms, seeds = _selected_matrix(
        matrix, only_arms=only_arms, only_seeds=only_seeds
    )
    model = matrix["model"]
    vector_dir = Path(matrix.get("vector_dir", "data/vectors"))
    if not vector_dir.is_absolute():
        vector_dir = (repo_root / vector_dir).resolve()

    specs: list[RunSpec] = []
    for arm in arms:
        arm_payload = matrix["arms"][arm]
        steering_override = {
            "active_traits": arm_payload.get("active_traits", ["E", "A", "C"]),
            "coefficients": arm_payload.get("coefficients", {}),
        }
        if arm_payload["steering_mode"] == "disabled":
            steering_override["enabled"] = False
        arm_override = {"steering": steering_override}
        for seed in seeds:
            run_id = f"{matrix['study_id']}-{arm}-s{seed}"
            run_dir = output_root / "runs" / arm / f"seed-{seed}"
            config = _deep_merge(base_config, matrix.get("common_overrides") or {})
            config = _deep_merge(config, arm_override)
            study_metadata: dict[str, Any] = {
                "study_id": matrix["study_id"],
                "arm": arm,
                "seed": seed,
                "steering_mode": arm_payload["steering_mode"],
                "placebo_algorithm": (
                    "trait_label_derangement_v1"
                    if arm_payload["steering_mode"] == "placebo"
                    else None
                ),
                "matrix_path": str(matrix_path.resolve()),
            }
            if arm_payload["steering_mode"] == "placebo":
                study_metadata["placebo_vector_mapping"] = _placebo_mapping(
                    steering_override["active_traits"], seed
                )
            config.update(
                {
                    "run_id": run_id,
                    "population": int(matrix["population"]),
                    "steps": int(matrix["steps"]),
                    "seed": seed,
                    "max_events_per_tick": int(
                        matrix.get("max_events_per_tick", config.get("max_events_per_tick", 16))
                    ),
                    "model_name": model["id"],
                    "model_revision": model["revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                    "study": study_metadata,
                }
            )
            _normalize_config_paths(config, repo_root)
            config["logging"] = {
                "db_url": None,
                "parquet_dir": str((run_dir / "outputs").resolve()),
            }
            optimization = config.setdefault("optimization", {})
            optimization["offload_folder"] = str((run_dir / "offload").resolve())
            specs.append(
                RunSpec(
                    study_id=matrix["study_id"],
                    arm=arm,
                    seed=seed,
                    steering_mode=arm_payload["steering_mode"],
                    run_id=run_id,
                    run_dir=run_dir,
                    config=config,
                    environment=matrix.get("environment", "research"),
                    difficulty=int(matrix.get("difficulty", 3)),
                    max_events_per_tick=int(config["max_events_per_tick"]),
                    vector_dir=vector_dir,
                )
            )
    return specs


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable study artifact differs from requested content: {path}")
        return
    _write_atomic(path, payload)


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def materialize_spec(
    spec: RunSpec,
    *,
    matrix_path: Path,
    base_config_path: Path,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config_bytes = _yaml_bytes(spec.config)
    config_path = spec.run_dir / "config.yaml"
    manifest_path = spec.run_dir / "run-manifest.json"
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "immutable_society_study_run",
        "study_id": spec.study_id,
        "run_id": spec.run_id,
        "arm": spec.arm,
        "seed": spec.seed,
        "steering_mode": spec.steering_mode,
        "placebo_algorithm": spec.config["study"].get("placebo_algorithm"),
        "placebo_vector_mapping": spec.config["study"].get("placebo_vector_mapping"),
        "execution_order": "sequential",
        "analysis_unit": "simulation_run",
        "population": int(spec.config["population"]),
        "steps": int(spec.config["steps"]),
        "config_sha256": sha256_bytes(config_bytes),
        "matrix_sha256": sha256_file(matrix_path),
        "base_config_sha256": sha256_file(base_config_path),
        "runner_sha256": sha256_file(script_path),
        "git_commit": _git_commit(repo_root),
    }
    _write_immutable(config_path, config_bytes)
    _write_immutable(manifest_path, _json_bytes(manifest))
    return manifest


def estimate_plan(
    specs: Sequence[RunSpec], matrix: dict[str, Any], *, mock_model: bool
) -> dict[str, Any]:
    estimate = matrix["estimation"]
    input_tokens = float(estimate["mean_input_tokens_per_generation"])
    output_tokens = float(estimate["mean_output_tokens_per_generation"])
    seconds = float(estimate["seconds_per_generation"])
    rows = []
    for spec in specs:
        max_participants_per_tick = min(
            int(spec.config["population"]), 3 * spec.max_events_per_tick
        )
        generations = int(spec.config["steps"]) * max_participants_per_tick
        tokens = generations * (input_tokens + output_tokens)
        rows.append(
            {
                "run_id": spec.run_id,
                "arm": spec.arm,
                "seed": spec.seed,
                "estimated_generations_upper_bound": generations,
                "estimated_tokens": int(tokens),
                "estimated_gpu_hours": 0.0 if mock_model else generations * seconds / 3600,
            }
        )
    return {
        "study_id": matrix["study_id"],
        "execution_mode": "sequential",
        "mock_model": mock_model,
        "assumptions": estimate,
        "n_runs": len(rows),
        "estimated_generations_upper_bound": sum(
            row["estimated_generations_upper_bound"] for row in rows
        ),
        "estimated_tokens": sum(row["estimated_tokens"] for row in rows),
        "estimated_gpu_hours": sum(row["estimated_gpu_hours"] for row in rows),
        "runs": rows,
    }


def enforce_real_run_budget(
    plan: dict[str, Any],
    *,
    mock_model: bool,
    max_estimated_gpu_hours: float | None,
    hourly_rate_usd: float | None,
    max_estimated_cost_usd: float | None,
) -> None:
    """Require explicit, sufficient time and optional dollar caps for real runs."""
    if mock_model:
        return
    if max_estimated_gpu_hours is None:
        raise ValueError(
            "real execution requires --max-estimated-gpu-hours; the explicit "
            "--execute flag alone is not a sufficient budget authorization"
        )
    if max_estimated_gpu_hours < 0:
        raise ValueError("--max-estimated-gpu-hours must be non-negative")
    estimated_hours = float(plan["estimated_gpu_hours"])
    if estimated_hours > max_estimated_gpu_hours:
        raise ValueError(
            f"estimated {estimated_hours:.3f} GPU-hours exceeds the authorized "
            f"cap of {max_estimated_gpu_hours:.3f}"
        )
    if (hourly_rate_usd is None) != (max_estimated_cost_usd is None):
        raise ValueError(
            "--hourly-rate-usd and --max-estimated-cost-usd must be supplied together"
        )
    if hourly_rate_usd is None:
        return
    if hourly_rate_usd < 0 or max_estimated_cost_usd is None or max_estimated_cost_usd < 0:
        raise ValueError("hourly rate and estimated cost cap must be non-negative")
    estimated_cost = estimated_hours * hourly_rate_usd
    plan["hourly_rate_usd"] = hourly_rate_usd
    plan["estimated_cost_usd"] = estimated_cost
    plan["max_estimated_cost_usd"] = max_estimated_cost_usd
    if estimated_cost > max_estimated_cost_usd:
        raise ValueError(
            f"estimated ${estimated_cost:.2f} cost exceeds the authorized "
            f"cap of ${max_estimated_cost_usd:.2f}"
        )


def _command_for_spec(spec: RunSpec, *, mock_model: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "orchestrator.cli",
        str(spec.run_dir / "config.yaml"),
        "--env",
        spec.environment,
        "--difficulty",
        str(spec.difficulty),
        "--max-events",
        str(spec.max_events_per_tick),
        "--steering-mode",
        spec.steering_mode,
        "--vector-dir",
        str(spec.vector_dir),
    ]
    if mock_model:
        command.append("--mock-model")
    return command


def _completion_valid(spec: RunSpec, manifest: dict[str, Any]) -> bool:
    path = spec.run_dir / "completed.json"
    if not path.is_file():
        return False
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        completion.get("run_id") == spec.run_id
        and completion.get("config_sha256") == manifest["config_sha256"]
        and completion.get("run_manifest_sha256")
        == sha256_file(spec.run_dir / "run-manifest.json")
    )


def _existing_parquet(spec: RunSpec) -> list[Path]:
    return sorted((spec.run_dir / "outputs").rglob("*.parquet"))


RunCommand = Callable[[list[str], Path, Path], int]


def _default_run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def execute_specs(
    specs: Sequence[RunSpec],
    manifests: dict[str, dict[str, Any]],
    *,
    mock_model: bool,
    resume: bool,
    repo_root: Path = PROJECT_ROOT,
    run_command: RunCommand = _default_run_command,
) -> dict[str, list[str]]:
    completed: list[str] = []
    skipped: list[str] = []
    for spec in specs:  # Intentionally sequential: one shared GPU is the default design.
        manifest = manifests[spec.run_id]
        if _completion_valid(spec, manifest):
            if resume:
                skipped.append(spec.run_id)
                continue
            raise FileExistsError(
                f"run is already complete; pass --resume to reuse it: {spec.run_id}"
            )
        existing = _existing_parquet(spec)
        if existing:
            raise RuntimeError(
                f"run {spec.run_id} has outputs but no valid completion marker; refusing "
                "to append or overwrite them. Inspect the run before retrying."
            )
        logs = sorted(spec.run_dir.glob("runner.attempt-*.log"))
        log_path = spec.run_dir / f"runner.attempt-{len(logs) + 1:03d}.log"
        command = _command_for_spec(spec, mock_model=mock_model)
        returncode = run_command(command, repo_root, log_path)
        if returncode != 0:
            raise RuntimeError(
                f"run {spec.run_id} failed with exit code {returncode}; see {log_path}"
            )
        output_files = _existing_parquet(spec)
        if not output_files:
            raise RuntimeError(f"run {spec.run_id} exited successfully but wrote no Parquet outputs")
        completion = {
            "schema_version": "1.0",
            "run_id": spec.run_id,
            "config_sha256": manifest["config_sha256"],
            "run_manifest_sha256": sha256_file(spec.run_dir / "run-manifest.json"),
            "mock_model": mock_model,
            "parquet_file_count": len(output_files),
        }
        _write_immutable(spec.run_dir / "completed.json", _json_bytes(completion))
        completed.append(spec.run_id)
    return {"completed": completed, "skipped": skipped}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/society_study")
    )
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--dry-run", action="store_true", help="materialize and estimate only")
    gate.add_argument("--execute", action="store_true", help="explicitly permit simulation execution")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip valid completed runs")
    parser.add_argument(
        "--max-estimated-gpu-hours",
        type=float,
        help="mandatory upper-bound authorization for --execute with a real model",
    )
    parser.add_argument(
        "--hourly-rate-usd",
        type=float,
        help="optional quoted GPU hourly rate; requires --max-estimated-cost-usd",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        help="optional dollar authorization cap; requires --hourly-rate-usd",
    )
    parser.add_argument("--only-arm", action="append")
    parser.add_argument("--only-seed", action="append", type=int)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    matrix_path = args.matrix.resolve()
    matrix = load_matrix(matrix_path)
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = (PROJECT_ROOT / base_path).resolve()
    specs = build_run_specs(
        matrix,
        matrix_path=matrix_path,
        output_root=args.output_root,
        only_arms=set(args.only_arm) if args.only_arm else None,
        only_seeds=set(args.only_seed) if args.only_seed else None,
    )
    plan = estimate_plan(specs, matrix, mock_model=args.mock_model)
    if args.execute:
        enforce_real_run_budget(
            plan,
            mock_model=args.mock_model,
            max_estimated_gpu_hours=args.max_estimated_gpu_hours,
            hourly_rate_usd=args.hourly_rate_usd,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
        )
    elif args.hourly_rate_usd is not None:
        if args.hourly_rate_usd < 0:
            raise ValueError("--hourly-rate-usd must be non-negative")
        plan["hourly_rate_usd"] = args.hourly_rate_usd
        plan["estimated_cost_usd"] = plan["estimated_gpu_hours"] * args.hourly_rate_usd
    manifests = {
        spec.run_id: materialize_spec(
            spec,
            matrix_path=matrix_path,
            base_config_path=base_path,
        )
        for spec in specs
    }
    _write_atomic(args.output_root.resolve() / "study-plan.json", _json_bytes(plan))
    print(json.dumps({key: value for key, value in plan.items() if key != "runs"}, indent=2))
    if args.dry_run:
        return 0
    result = execute_specs(
        specs,
        manifests,
        mock_model=args.mock_model,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

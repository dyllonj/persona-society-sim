#!/usr/bin/env python3
"""CI harness that sweeps seeds/alphas around the steering evaluation scripts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from steering.ci_checks import (
    TraitCurvePoint,
    TraitDirectionality,
    extract_trait_rows,
    validate_cosine_stability,
    validate_directionality,
    validate_monotonic_logprobs,
)
from steering.vector_store import VectorStore


DEFAULT_STRENGTHS = (0.5, 1.0)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_TRAITS = ("extraversion", "agreeableness", "conscientiousness")


@dataclass(frozen=True)
class TraitSpec:
    code: str
    name: str
    vector_store_id: str
    layers: List[int]


def _format_alpha(alpha: float) -> str:
    return str(alpha).replace("-", "m").replace(".", "p")


def _load_trait_specs(config_path: Path, only: Sequence[str] | None) -> List[TraitSpec]:
    config = yaml.safe_load(config_path.read_text()) or {}
    defaults = config.get("defaults") or {}
    default_layers = defaults.get("layers") or []
    selected = set(item.lower() for item in only) if only else None

    specs: List[TraitSpec] = []
    for trait_code, payload in (config.get("traits") or {}).items():
        name = (payload.get("name") or trait_code).lower()
        if selected and name not in selected and trait_code.lower() not in selected:
            continue
        layers = payload.get("layers") or default_layers
        specs.append(
            TraitSpec(
                code=trait_code,
                name=name,
                vector_store_id=payload.get("vector_store_id") or trait_code,
                layers=list(layers),
            )
        )

    if not specs:
        raise ValueError("No trait definitions matched the requested filter")
    return specs


def _run_command(cmd: Sequence[str], env: Dict[str, str]) -> None:
    subprocess.run(cmd, env=env, check=True)


def _load_vectors(
    *,
    vector_roots: Sequence[Tuple[int, Path]],
    trait_specs: Sequence[TraitSpec],
) -> Dict[str, Dict[int, List[Tuple[int, Sequence[float]]]]]:
    grouped: Dict[str, Dict[int, List[Tuple[int, Sequence[float]]]]] = {}
    for seed, root in vector_roots:
        store = VectorStore(root)
        for spec in trait_specs:
            bundle = store.load(spec.vector_store_id, layers=spec.layers)
            layer_map = grouped.setdefault(spec.code, {})
            for layer_id, vector in bundle.vectors.items():
                layer_map.setdefault(layer_id, []).append((seed, vector))
    return grouped


def _parse_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing steering eval report at {path}")
    return json.loads(path.read_text())


def _validate_markdown(path: Path, trait_count: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected Markdown output alongside {path.with_suffix('.json')}")
    content = path.read_text().strip()
    if not content:
        raise ValueError(f"Markdown report {path} is empty")
    rows = [line for line in content.splitlines() if line.startswith("|")]
    data_rows = [line for line in rows if not line.startswith("| Trait ") and "---" not in line]
    if len(data_rows) < trait_count:
        raise ValueError(
            f"Markdown report {path} is missing trait rows (expected {trait_count}, found {len(data_rows)})"
        )


def _collect_directionality(
    *,
    report: dict,
    report_path: Path,
    seed: int,
    alpha: float,
) -> List[TraitDirectionality]:
    rows = extract_trait_rows(report)
    results: List[TraitDirectionality] = []
    for row in rows:
        results.append(
            TraitDirectionality(
                trait=row.get("trait_name") or row.get("trait_code") or "unknown",
                seed=seed,
                alpha=alpha,
                sign_consistency=float(row.get("sign_consistency", 0.0)),
                directional_improvement=float(row.get("directional_improvement", 0.0)),
                source=report_path,
            )
        )
    return results


def _collect_logprob_curves(
    *, report: dict, report_path: Path, seed: int, alpha: float
) -> List[TraitCurvePoint]:
    rows = extract_trait_rows(report)
    results: List[TraitCurvePoint] = []
    for row in rows:
        logprob = row.get("logprob_gap", {})
        results.append(
            TraitCurvePoint(
                trait=row.get("trait_name") or row.get("trait_code") or "unknown",
                seed=seed,
                alpha=alpha,
                logprob_gap_delta=float(logprob.get("delta", 0.0)),
                source=report_path,
            )
        )
    return results


def _build_env(
    base: Dict[str, str],
    *,
    vector_root: Path,
    artifact_dir: Path,
    alpha: float,
    args: argparse.Namespace,
) -> Dict[str, str]:
    env = base.copy()
    env.update(
        {
            "VECTOR_ROOT": str(vector_root),
            "ARTIFACT_DIR": str(artifact_dir),
            "STEERING_ALPHA": str(alpha),
            "DELTA_THRESHOLD": str(args.delta_threshold),
            "SIGN_THRESHOLD": str(args.sign_threshold),
            "PROMPT_DIR": str(args.prompt_dir),
            "SKIP_VECTOR_REGEN": "1",
        }
    )
    if args.traits:
        env["TRAITS"] = " ".join(args.traits)
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run steering vector sweeps with cosine/directionality gates.",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--vector-metadata", type=Path, default=Path("configs/steering.layers.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/steering_eval"))
    parser.add_argument("--compute-script", type=Path, default=Path("scripts/compute_vectors.sh"))
    parser.add_argument("--eval-script", type=Path, default=Path("scripts/eval_vectors.sh"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--strengths", type=float, nargs="*", default=list(DEFAULT_STRENGTHS))
    parser.add_argument("--traits", nargs="*", default=list(DEFAULT_TRAITS))
    parser.add_argument("--cosine-threshold", type=float, default=0.995)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-4)
    parser.add_argument("--directional-threshold", type=float, default=0.6)
    parser.add_argument("--sign-threshold", type=float, default=0.55)
    parser.add_argument("--delta-threshold", type=float, default=0.1)
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--vector-root-override", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    trait_specs = _load_trait_specs(args.vector_metadata, args.traits)

    directionality_points: List[TraitDirectionality] = []
    curve_points: List[TraitCurvePoint] = []
    vector_roots: List[Tuple[int, Path]] = []

    for seed in args.seeds:
        vector_root = (
            args.vector_root_override.resolve()
            if args.vector_root_override
            else artifact_root / f"vectors_seed{seed}"
        )
        vector_roots.append((seed, vector_root))
        base_env = os.environ.copy()
        base_env.update(
            {
                "MODEL_NAME": args.model,
                "VECTOR_METADATA": str(args.vector_metadata),
                "PYTHONHASHSEED": str(seed),
                "HF_SEED": str(seed),
            }
        )

        compute_env = base_env.copy()
        compute_env.update({"VECTOR_ROOT": str(vector_root)})

        if not args.dry_run:
            _run_command([str(args.compute_script)], compute_env)

        for alpha in args.strengths:
            artifact_dir = artifact_root / f"seed_{seed}" / f"alpha_{_format_alpha(alpha)}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            eval_env = _build_env(
                base_env,
                vector_root=vector_root,
                artifact_dir=artifact_dir,
                alpha=alpha,
                args=args,
            )

            if not args.dry_run:
                _run_command([str(args.eval_script)], eval_env)

            report_path = artifact_dir / "report.json"
            markdown_path = artifact_dir / "report.md"

            report = _parse_report(report_path)
            _validate_markdown(markdown_path, trait_count=len(report.get("traits", [])))

            directionality_points.extend(
                _collect_directionality(
                    report=report, report_path=report_path, seed=seed, alpha=alpha
                )
            )
            curve_points.extend(
                _collect_logprob_curves(
                    report=report, report_path=report_path, seed=seed, alpha=alpha
                )
            )

            if report.get("failures"):
                raise SystemExit(
                    f"Vector evaluation at seed={seed}, alpha={alpha} reported failures: {report['failures']}"
                )

    vector_sets = _load_vectors(vector_roots=vector_roots, trait_specs=trait_specs)

    failure_messages = []
    failure_messages.extend(
        validate_cosine_stability(vector_sets, threshold=args.cosine_threshold)
    )
    failure_messages.extend(
        validate_directionality(
            directionality_points,
            sign_threshold=args.sign_threshold,
            directional_threshold=args.directional_threshold,
        )
    )
    failure_messages.extend(
        validate_monotonic_logprobs(
            curve_points, tolerance=max(args.logprob_tolerance, 0.0)
        )
    )

    if failure_messages:
        formatted = "\n- " + "\n- ".join(failure_messages)
        raise SystemExit(f"Steering CI harness detected failures:{formatted}")

    print("Steering CI harness completed successfully.")


if __name__ == "__main__":
    main()

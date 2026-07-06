"""Steerability smoke checks before expensive vector extraction runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

GateStatus = str


@dataclass(frozen=True)
class DirectionalAgreement:
    """Projection-sign summary for a batch of activation differences."""

    count: int
    aligned: int
    neutral: int
    opposed: int
    agreement: float
    mean_projection: float
    min_projection: float
    max_projection: float


@dataclass(frozen=True)
class SmokeGate:
    """PASS/WARN/FAIL result for a steerability smoke batch."""

    status: GateStatus
    summary: DirectionalAgreement
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LayerSteerabilityScore:
    """Preliminary steerability score for one trait/layer pair."""

    trait: str
    layer: int
    count: int
    preliminary_steerability: float
    status: GateStatus
    reasons: tuple[str, ...]


def _as_float_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Activation diff entries must not be empty")
    return array


def project_activation_diff(
    activation_diff: float | Sequence[float],
    reference_vector: Sequence[float] | None = None,
) -> float:
    """Project one activation diff onto a reference direction.

    Scalar diffs are already signed projections. Vector diffs require a
    reference vector and use a dot product.
    """

    diff = _as_float_array(activation_diff)
    if diff.ndim == 0 or diff.size == 1:
        return float(diff.reshape(-1)[0])
    if reference_vector is None:
        raise ValueError("Vector activation diffs require a reference_vector")
    reference = _as_float_array(reference_vector)
    if reference.shape != diff.shape:
        raise ValueError(
            f"reference_vector shape {reference.shape} does not match activation diff shape {diff.shape}"
        )
    return float(np.dot(diff.reshape(-1), reference.reshape(-1)))


def directional_agreement(
    activation_diffs: Iterable[float | Sequence[float]],
    reference_vector: Sequence[float] | None = None,
    *,
    deadband: float = 0.0,
) -> DirectionalAgreement:
    """Return the fraction of activation diffs aligned with the expected direction."""

    projections = [
        project_activation_diff(diff, reference_vector) for diff in activation_diffs
    ]
    if not projections:
        raise ValueError("At least one activation diff is required")

    aligned = sum(1 for value in projections if value > deadband)
    opposed = sum(1 for value in projections if value < -deadband)
    neutral = len(projections) - aligned - opposed
    count = len(projections)
    return DirectionalAgreement(
        count=count,
        aligned=aligned,
        neutral=neutral,
        opposed=opposed,
        agreement=aligned / count,
        mean_projection=float(np.mean(projections)),
        min_projection=float(np.min(projections)),
        max_projection=float(np.max(projections)),
    )


def mean_cosine_directional_agreement(activation_diffs: Sequence[Sequence[float]]) -> float:
    """Mean cosine similarity of individual diffs with the mean direction."""

    matrix = np.asarray(activation_diffs, dtype=np.float64)
    if matrix.size == 0:
        raise ValueError("At least one activation diff is required")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    mean_vec = matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    if mean_norm == 0.0:
        return 0.0
    cosines: list[float] = []
    for row in matrix:
        row_norm = float(np.linalg.norm(row))
        if row_norm == 0.0:
            cosines.append(0.0)
        else:
            cosines.append(float(np.dot(row, mean_vec) / (row_norm * mean_norm)))
    return float(np.mean(cosines)) if cosines else 0.0


def gate_preliminary_steerability(
    *,
    trait: str,
    layer: int,
    activation_diffs: Sequence[Sequence[float]],
    pass_threshold: float = 0.3,
    warn_threshold: float = 0.2,
    min_count: int = 5,
) -> LayerSteerabilityScore:
    """Gate one trait/layer by mean cosine agreement against the mean diff."""

    if not 0.0 <= warn_threshold <= pass_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= warn <= pass <= 1")
    count = len(activation_diffs)
    score = mean_cosine_directional_agreement(activation_diffs) if activation_diffs else 0.0
    reasons: list[str] = []
    if count < min_count:
        reasons.append(f"count {count} < min_count {min_count}")
    if score < warn_threshold:
        reasons.append(
            f"preliminary_steerability {score:.3f} < warn_threshold {warn_threshold:.3f}"
        )
    elif score < pass_threshold:
        reasons.append(
            f"preliminary_steerability {score:.3f} < pass_threshold {pass_threshold:.3f}"
        )

    if count < min_count or score < warn_threshold:
        status = "FAIL"
    elif score >= pass_threshold:
        status = "PASS"
    else:
        status = "WARN"
    return LayerSteerabilityScore(
        trait=trait,
        layer=layer,
        count=count,
        preliminary_steerability=score,
        status=status,
        reasons=tuple(reasons),
    )


def gate_directional_agreement(
    summary: DirectionalAgreement,
    *,
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.6,
    min_count: int = 1,
    min_mean_projection: float = 0.0,
) -> SmokeGate:
    """Convert a directional agreement summary into PASS/WARN/FAIL."""

    if not 0.0 <= warn_threshold <= pass_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= warn <= pass <= 1")

    reasons: list[str] = []
    if summary.count < min_count:
        reasons.append(f"count {summary.count} < min_count {min_count}")
    if summary.mean_projection < min_mean_projection:
        reasons.append(
            f"mean_projection {summary.mean_projection:.6f} < {min_mean_projection:.6f}"
        )
    if summary.agreement < warn_threshold:
        reasons.append(
            f"agreement {summary.agreement:.3f} < warn_threshold {warn_threshold:.3f}"
        )
    elif summary.agreement < pass_threshold:
        reasons.append(
            f"agreement {summary.agreement:.3f} < pass_threshold {pass_threshold:.3f}"
        )

    if summary.count < min_count or summary.agreement < warn_threshold:
        status = "FAIL"
    elif summary.agreement >= pass_threshold and summary.mean_projection >= min_mean_projection:
        status = "PASS"
    else:
        status = "WARN"

    return SmokeGate(status=status, summary=summary, reasons=tuple(reasons))


def gate_activation_diffs(
    activation_diffs: Iterable[float | Sequence[float]],
    reference_vector: Sequence[float] | None = None,
    *,
    deadband: float = 0.0,
    pass_threshold: float = 0.8,
    warn_threshold: float = 0.6,
    min_count: int = 1,
    min_mean_projection: float = 0.0,
) -> SmokeGate:
    """Compute directional agreement and return a PASS/WARN/FAIL gate."""

    summary = directional_agreement(
        activation_diffs, reference_vector, deadband=deadband
    )
    return gate_directional_agreement(
        summary,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
        min_count=min_count,
        min_mean_projection=min_mean_projection,
    )


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_diff(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    for key in ("activation_diff", "diff", "projection"):
        if key in entry:
            return entry[key]
    raise ValueError(
        "Diff records must contain one of: activation_diff, diff, projection"
    )


def load_activation_diff_file(
    path: Path,
) -> tuple[list[float | Sequence[float]], Sequence[float] | None]:
    """Load scalar or vector activation diffs from JSON/JSONL/NPY files."""

    if path.suffix == ".npy":
        array = np.load(path)
        if array.ndim == 0:
            return [float(array)], None
        return array.tolist(), None

    payload = _load_json_or_jsonl(path)
    reference_vector = None
    if isinstance(payload, Mapping):
        reference_vector = payload.get("reference_vector")
        entries = (
            payload.get("activation_diffs")
            or payload.get("diffs")
            or payload.get("records")
        )
        if entries is None:
            entries = [payload]
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ValueError("Activation diff payload must be a list or records object")
    return [_extract_diff(entry) for entry in entries], reference_vector


def _load_reference_vector(value: str | None, path: Path | None) -> Sequence[float] | None:
    if value and path:
        raise ValueError("Use only one of --reference-vector or --reference-vector-file")
    if path is not None:
        if path.suffix == ".npy":
            return np.load(path).tolist()
        return json.loads(path.read_text(encoding="utf-8"))
    if value:
        stripped = value.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        return [float(part.strip()) for part in stripped.split(",") if part.strip()]
    return None


def _coerce_trait_list(value: str | None, config_traits: Sequence[str]) -> list[str]:
    if not value:
        return list(config_traits)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    return selected or list(config_traits)


def _load_config_traits(config_path: Path) -> list[str]:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    traits = payload.get("traits") or {}
    if not isinstance(traits, Mapping) or not traits:
        raise ValueError(f"No traits found in {config_path}")
    return [str(trait) for trait in traits]


def _sample_prompts(path: Path, sample_pairs: int):
    from data.prompts.schema import load_prompt_items

    prompts = load_prompt_items(path)
    return prompts[: max(1, sample_pairs)]


def _select_smoke_layers(
    layers: Sequence[int],
    *,
    layer_step: int | None = None,
) -> tuple[int, ...]:
    unique_layers = tuple(sorted({int(layer) for layer in layers}))
    if not unique_layers:
        raise ValueError("At least one layer is required for smoke testing")
    if layer_step is None or layer_step <= 0:
        return unique_layers
    selected = [layer for layer in unique_layers if layer % layer_step == 0]
    return tuple(selected or unique_layers)


def run_model_smoke_test(
    *,
    config_path: Path,
    traits: Sequence[str] | None = None,
    model_name: str | None = None,
    sample_pairs: int = 10,
    layer_step: int | None = None,
    pass_threshold: float = 0.3,
    warn_threshold: float = 0.2,
    min_count: int = 5,
) -> dict[str, Any]:
    """Run a quick HF activation-diff smoke test from YAML prompt metadata."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering.compute_caa import (
        encode_answer_token,
        high_low_letters,
        resolve_caa_config,
        validate_model_config,
    )

    config_traits = _load_config_traits(config_path)
    trait_codes = list(traits or config_traits)
    resolved_configs = [
        resolve_caa_config(trait, config_path=config_path, model=model_name)
        for trait in trait_codes
    ]
    model_id = resolved_configs[0].model
    for resolved in resolved_configs[1:]:
        if resolved.model != model_id:
            raise ValueError("All smoke-test traits must resolve to the same model")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    validate_model_config(model, resolved_configs[0])
    model.eval()

    scores: list[LayerSteerabilityScore] = []
    for resolved in resolved_configs:
        validate_model_config(model, resolved)
        layers = _select_smoke_layers(resolved.layers, layer_step=layer_step)
        prompts = _sample_prompts(resolved.prompt_file, sample_pairs)
        per_layer: dict[int, list[list[float]]] = {layer: [] for layer in layers}
        for prompt in prompts:
            high_letter, low_letter = high_low_letters(prompt)
            high_state = encode_answer_token(
                model,
                tokenizer,
                prompt.question_text,
                high_letter,
                layers,
            )
            low_state = encode_answer_token(
                model,
                tokenizer,
                prompt.question_text,
                low_letter,
                layers,
            )
            for layer in layers:
                diff = (high_state[layer] - low_state[layer]).detach().cpu().numpy()
                per_layer[layer].append(diff.tolist())
        for layer, diffs in per_layer.items():
            scores.append(
                gate_preliminary_steerability(
                    trait=resolved.trait,
                    layer=layer,
                    activation_diffs=diffs,
                    pass_threshold=pass_threshold,
                    warn_threshold=warn_threshold,
                    min_count=min_count,
                )
            )

    trait_status: dict[str, str] = {}
    for trait in trait_codes:
        statuses = [score.status for score in scores if score.trait == trait]
        if not statuses:
            trait_status[trait] = "FAIL"
        elif "PASS" in statuses:
            trait_status[trait] = "PASS"
        elif "WARN" in statuses:
            trait_status[trait] = "WARN"
        else:
            trait_status[trait] = "FAIL"

    return {
        "model": model_id,
        "config": str(config_path),
        "sample_pairs": sample_pairs,
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "trait_status": trait_status,
        "scores": [asdict(score) for score in scores],
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate steering activation diffs before expensive extraction runs."
    )
    parser.add_argument("--diffs", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--traits", help="Comma-separated trait codes for --config mode")
    parser.add_argument("--model", help="Override defaults.model in --config mode")
    parser.add_argument("--sample-pairs", type=int, default=10)
    parser.add_argument("--layer-step", type=int)
    parser.add_argument("--reference-vector")
    parser.add_argument("--reference-vector-file", type=Path)
    parser.add_argument("--deadband", type=float, default=0.0)
    parser.add_argument("--pass-threshold", type=float, default=0.8)
    parser.add_argument("--warn-threshold", type=float, default=0.6)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--min-mean-projection", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if bool(args.diffs) == bool(args.config):
        raise ValueError("Provide exactly one of --diffs or --config")

    if args.config:
        config_traits = _load_config_traits(args.config)
        selected_traits = _coerce_trait_list(args.traits, config_traits)
        payload = run_model_smoke_test(
            config_path=args.config,
            traits=selected_traits,
            model_name=args.model,
            sample_pairs=args.sample_pairs,
            layer_step=args.layer_step,
            pass_threshold=args.pass_threshold,
            warn_threshold=args.warn_threshold,
            min_count=args.min_count,
        )
        exit_code = 1 if any(status == "FAIL" for status in payload["trait_status"].values()) else 0
    else:
        diffs, payload_reference = load_activation_diff_file(args.diffs)
        cli_reference = _load_reference_vector(
            args.reference_vector, args.reference_vector_file
        )
        reference_vector = cli_reference if cli_reference is not None else payload_reference
        gate = gate_activation_diffs(
            diffs,
            reference_vector,
            deadband=args.deadband,
            pass_threshold=args.pass_threshold,
            warn_threshold=args.warn_threshold,
            min_count=args.min_count,
            min_mean_projection=args.min_mean_projection,
        )
        payload = asdict(gate)
        exit_code = 1 if gate.status == "FAIL" else 0
    json_payload = json.dumps(payload, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_payload + "\n", encoding="utf-8")
    print(json_payload)
    return exit_code


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

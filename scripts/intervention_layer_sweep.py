"""Sweep actual steering interventions by layer/sign without reloading the model.

This complements ``steering.layer_sweep``.  That module scores whether held-out
activation differences align with stored vectors; this script tests the runtime
hook path used by evaluation and simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from steering.eval import (
    HFContrastScorer,
    _load_prompt_records,
    _resolve_prompt_path,
    _trait_to_dict,
    canonicalize_trait,
    evaluate_trait_dataset,
)
from steering.vector_store import VectorStore


def _parse_float_csv(value: str) -> List[float]:
    parsed: List[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if item:
            parsed.append(float(item))
    if not parsed:
        raise ValueError("Expected at least one alpha")
    return parsed


def _parse_layer_csv(value: str) -> Tuple[int, ...]:
    parsed = tuple(sorted({int(raw.strip()) for raw in value.split(",") if raw.strip()}))
    if not parsed:
        raise ValueError("Expected at least one layer")
    return parsed


def _dedupe_layer_sets(layer_sets: Iterable[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    seen = set()
    unique: List[Tuple[int, ...]] = []
    for layers in layer_sets:
        key = tuple(layers)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _load_metadata(metadata_root: Path, trait_code: str) -> dict:
    path = metadata_root / f"{trait_code}.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata for trait {trait_code}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_name(metadata: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("Pass --model or provide model_name in vector metadata")
    return str(model_name)


def _load_layer_tensors(
    metadata_root: Path, metadata: dict, *, apply_polarity: bool
) -> dict[int, torch.Tensor]:
    vector_store_id = metadata.get("vector_store_id")
    if not vector_store_id:
        raise ValueError("Vector metadata is missing vector_store_id")
    bundle = VectorStore(metadata_root).load(str(vector_store_id))
    vectors = bundle.calibrated_vectors() if apply_polarity else bundle.vectors
    return {
        int(layer_id): torch.tensor(array, dtype=torch.float32)
        for layer_id, array in vectors.items()
    }


def _set_active_layers(
    scorer: HFContrastScorer,
    trait_code: str,
    all_vectors: dict[int, torch.Tensor],
    layers: Sequence[int],
) -> None:
    active = {int(layer): all_vectors[int(layer)] for layer in layers}
    scorer.trait_vectors[trait_code] = active
    if scorer.controller is not None:
        scorer.controller.trait_vectors[trait_code] = active
        scorer.controller._batched_cache.clear()  # pylint: disable=protected-access
        scorer.controller.clear_prompt_metadata()


def _build_layer_sets(
    layers: Sequence[int],
    preferred_layers: Sequence[int],
    extra_layer_sets: Sequence[str] | None,
    include_preferred: bool,
) -> List[Tuple[int, ...]]:
    candidates: List[Tuple[int, ...]] = [(int(layer),) for layer in sorted(layers)]
    if include_preferred and preferred_layers:
        candidates.append(tuple(sorted({int(layer) for layer in preferred_layers})))
    for raw in extra_layer_sets or []:
        candidates.append(_parse_layer_csv(raw))
    return _dedupe_layer_sets(candidates)


def _markdown(report: dict) -> str:
    lines = [
        "# Intervention Layer Sweep",
        "",
        f"Model: `{report['model']}`",
        f"Trait: `{report['trait_name']}` (`{report['trait_code']}`)",
        f"Prompt file: `{report['prompt_path']}`",
        "",
        "| Layers | Alpha | Gap Delta | Anti | Directional | Accuracy Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        trait = row["trait"]
        lines.append(
            "| {layers} | {alpha:.3f} | {gap:.4f} | {anti:.3f} | {directional:.3f} | {acc:.3f} |".format(
                layers=",".join(str(layer) for layer in row["layers"]),
                alpha=row["alpha"],
                gap=trait["logprob_gap"]["delta"],
                anti=trait["anti_steerable_fraction"],
                directional=trait["directional_improvement"],
                acc=trait["accuracy"]["delta"],
            )
        )
    best = report.get("best_by_gap_delta")
    if best:
        trait = best["trait"]
        lines.extend(
            [
                "",
                "## Best Gap Delta",
                "",
                "- Layers: `{}`".format(",".join(str(layer) for layer in best["layers"])),
                f"- Alpha: `{best['alpha']}`",
                f"- Logprob gap delta: `{trait['logprob_gap']['delta']:.4f}`",
                f"- Anti-steerable fraction: `{trait['anti_steerable_fraction']:.3f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-hook layer/sign sweep for one steering trait."
    )
    parser.add_argument("trait", help="Trait identifier, e.g. E or extraversion")
    parser.add_argument("--metadata-root", type=Path, default=Path("data/vectors"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--prompt-override", type=Path)
    parser.add_argument("--eval-suffix", default="_eval")
    parser.add_argument("--model")
    parser.add_argument("--alphas", default="-3,-2,-1,-0.5,0.5,1,2,3")
    parser.add_argument(
        "--layer-set",
        action="append",
        help="Additional comma-separated layer set to test, e.g. 12,36",
    )
    parser.add_argument(
        "--include-preferred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also test metadata preferred_layers as a combined intervention.",
    )
    parser.add_argument(
        "--raw-vectors",
        action="store_true",
        help="Ignore metadata polarity and test raw extracted vectors.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/steering_eval/intervention_layer_sweep.json"),
    )
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    trait_name, trait_code = canonicalize_trait(args.trait)
    metadata = _load_metadata(args.metadata_root, trait_code)
    model_name = _resolve_model_name(metadata, args.model)
    prompt_path = args.prompt_override or _resolve_prompt_path(
        trait_name, args.prompt_dir, {}, args.eval_suffix
    )
    prompts = _load_prompt_records(prompt_path)
    all_vectors = _load_layer_tensors(
        args.metadata_root, metadata, apply_polarity=not args.raw_vectors
    )
    preferred_layers = metadata.get("preferred_layers") or []
    layer_sets = _build_layer_sets(
        sorted(all_vectors),
        preferred_layers,
        args.layer_set,
        args.include_preferred,
    )
    for layers in layer_sets:
        missing = [layer for layer in layers if layer not in all_vectors]
        if missing:
            raise ValueError(f"Layer set {layers} includes missing vectors {missing}")

    scorer = HFContrastScorer(model_name, {trait_code: dict(all_vectors)})
    results = []
    try:
        for layers in layer_sets:
            _set_active_layers(scorer, trait_code, all_vectors, layers)
            for alpha in _parse_float_csv(args.alphas):
                evaluation = evaluate_trait_dataset(
                    trait_name,
                    trait_code,
                    prompt_path,
                    prompts,
                    scorer,
                    alpha=alpha,
                    vector_store_id=str(metadata.get("vector_store_id") or trait_code),
                    metadata_path=args.metadata_root / f"{trait_code}.meta.json",
                )
                results.append(
                    {
                        "layers": list(layers),
                        "alpha": alpha,
                        "trait": _trait_to_dict(evaluation),
                    }
                )
    finally:
        scorer.close()

    best = max(
        results,
        key=lambda row: (
            row["trait"]["logprob_gap"]["delta"],
            -row["trait"]["anti_steerable_fraction"],
        ),
        default=None,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "trait_name": trait_name,
        "trait_code": trait_code,
        "prompt_path": str(prompt_path),
        "metadata_path": str(args.metadata_root / f"{trait_code}.meta.json"),
        "vector_store_id": metadata.get("vector_store_id") or trait_code,
        "polarity": metadata.get("polarity", 1.0),
        "used_raw_vectors": bool(args.raw_vectors),
        "preferred_layers": preferred_layers,
        "tested_layer_sets": [list(layers) for layers in layer_sets],
        "alphas": _parse_float_csv(args.alphas),
        "results": results,
        "best_by_gap_delta": best,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_output = args.markdown_output
    if markdown_output is None:
        markdown_output = args.json_output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {markdown_output}")
    if best:
        trait = best["trait"]
        print(
            "Best gap delta: layers={} alpha={} gap_delta={:.4f} anti={:.3f}".format(
                best["layers"],
                best["alpha"],
                trait["logprob_gap"]["delta"],
                trait["anti_steerable_fraction"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

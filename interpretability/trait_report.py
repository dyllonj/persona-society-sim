from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import jlens
import numpy as np
import torch
import yaml

try:  # Supports both `python -m` and direct script execution.
    from .common import sha256_file, sha256_json
except ImportError:  # pragma: no cover - direct script execution
    from common import sha256_file, sha256_json  # type: ignore


SCHEMA_VERSION = "1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transport configured steering vectors through a Jacobian Lens and report "
            "cross-layer trait interactions"
        )
    )
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--lens-manifest", type=Path, required=True)
    parser.add_argument("--vector-metadata", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="write <prefix>.json and <prefix>.md",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        default=[],
        metavar="TRAIT=VALUE",
        help="alpha for the combined-effect report; unspecified traits use 1.0",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--skip-vocab",
        action="store_true",
        help="do not load the Hugging Face model; suitable for CPU/offline analysis",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless the lens covers every configured trait/layer component",
    )
    return parser.parse_args()


def _resolve_vector_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    candidates = [path, root / path.name] if path.is_absolute() else [root / path, root / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"steering vector not found: {recorded}; root={root}")


def _load_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing steering vector index: {path}")
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            key = (str(payload["vector_store_id"]), int(payload["layer_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number} has an invalid vector identity") from exc
        previous = entries.get(key)
        if previous is not None:
            stable_fields = ("trait", "method", "vector_path", "train_set_hash", "eval_set_hash")
            if any(previous.get(field) != payload.get(field) for field in stable_fields):
                raise ValueError(f"conflicting duplicate vector index entry for {key}")
        entries[key] = payload
    return entries


def _load_lens(lens_path: Path, manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(lens_path)
    if actual_hash != manifest.get("lens_sha256"):
        raise ValueError(
            f"lens hash mismatch: manifest={manifest.get('lens_sha256')}, actual={actual_hash}"
        )
    lens = jlens.JacobianLens.load(str(lens_path))
    if int(manifest.get("d_model", -1)) != int(lens.d_model):
        raise ValueError(
            f"lens/manifest width mismatch: lens={lens.d_model}, "
            f"manifest={manifest.get('d_model')}"
        )
    manifest_layers = sorted(int(layer) for layer in manifest.get("source_layers", []))
    if manifest_layers != list(lens.source_layers):
        raise ValueError(
            f"lens/manifest source-layer mismatch: lens={lens.source_layers}, "
            f"manifest={manifest_layers}"
        )
    for layer in lens.source_layers:
        jacobian = lens.jacobians[layer]
        expected_shape = (lens.d_model, lens.d_model)
        if tuple(jacobian.shape) != expected_shape:
            raise ValueError(
                f"Jacobian shape mismatch at layer {layer}: {tuple(jacobian.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(jacobian).all():
            raise ValueError(f"Jacobian at layer {layer} contains non-finite values")
    return lens, manifest


def _load_vectors(
    metadata_path: Path,
    manifest: dict[str, Any],
    *,
    expected_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    defaults = config.get("defaults") or {}
    configured_model = defaults.get("model")
    if configured_model != manifest.get("model_id"):
        raise ValueError(
            f"vector/lens model mismatch: vectors={configured_model!r}, "
            f"lens={manifest.get('model_id')!r}"
        )
    configured_layers = defaults.get("num_hidden_layers")
    if configured_layers is not None and int(configured_layers) != int(manifest.get("n_layers")):
        raise ValueError(
            f"vector/lens layer-count mismatch: vectors={configured_layers}, "
            f"lens={manifest.get('n_layers')}"
        )

    vector_root_value = config.get("vector_root")
    if not vector_root_value:
        raise ValueError(f"{metadata_path} does not define vector_root")
    vector_root = (metadata_path.parent / str(vector_root_value)).resolve()
    index_path = vector_root / "index.jsonl"
    entries = _load_index(index_path)

    components: list[dict[str, Any]] = []
    traits = config.get("traits") or {}
    if not traits:
        raise ValueError(f"{metadata_path} does not configure any traits")
    for trait, trait_config in traits.items():
        vector_id = str(trait_config["vector_store_id"])
        polarity = float(trait_config.get("polarity", 1.0))
        if polarity not in (-1.0, 1.0):
            raise ValueError(f"invalid polarity for trait {trait}: {polarity}")
        for layer_value in trait_config.get("layers") or []:
            layer = int(layer_value)
            entry = entries.get((vector_id, layer))
            if entry is None:
                raise ValueError(f"vector index lacks {vector_id} layer {layer}")
            if str(entry.get("trait")) != str(trait):
                raise ValueError(
                    f"vector index trait mismatch for {vector_id} layer {layer}: "
                    f"metadata={trait}, index={entry.get('trait')}"
                )
            vector_path = _resolve_vector_path(vector_root, str(entry["vector_path"]))
            array = np.load(vector_path, allow_pickle=False)
            if array.ndim != 1 or int(array.shape[0]) != expected_width:
                raise ValueError(
                    f"vector shape mismatch for {trait}@{layer}: {array.shape}, "
                    f"expected ({expected_width},)"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"vector {trait}@{layer} contains non-finite values")
            vector = torch.from_numpy(array.astype(np.float32, copy=False)) * polarity
            if float(torch.linalg.vector_norm(vector)) == 0.0:
                raise ValueError(f"vector {trait}@{layer} has zero norm")
            components.append(
                {
                    "label": f"{trait}@{layer}",
                    "trait": str(trait),
                    "trait_name": str(trait_config.get("name") or trait),
                    "layer": layer,
                    "vector_store_id": vector_id,
                    "polarity": polarity,
                    "vector": vector,
                    "vector_path": vector_path,
                    "vector_sha256": sha256_file(vector_path),
                    "index_entry": entry,
                }
            )
    components.sort(key=lambda component: (component["trait"], component["layer"]))
    provenance = {
        "vector_metadata_path": str(metadata_path.resolve()),
        "vector_metadata_sha256": sha256_file(metadata_path),
        "vector_index_path": str(index_path.resolve()),
        "vector_index_sha256": sha256_file(index_path),
        "vector_root": str(vector_root),
    }
    return components, provenance


def _parse_alphas(values: list[str], traits: set[str]) -> dict[str, float]:
    alphas = {trait: 1.0 for trait in traits}
    seen: set[str] = set()
    for item in values:
        trait, separator, raw_value = item.partition("=")
        trait = trait.strip()
        if not separator or not trait or trait not in traits:
            raise ValueError(f"invalid --alpha {item!r}; expected one of {sorted(traits)}=VALUE")
        if trait in seen:
            raise ValueError(f"duplicate --alpha for trait {trait}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"alpha for trait {trait} must be finite")
        alphas[trait] = value
        seen.add(trait)
    return dict(sorted(alphas.items()))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _matrix(vectors: list[torch.Tensor]) -> tuple[list[list[float]], list[list[float | None]]]:
    stacked = torch.stack(vectors).float()
    gram_tensor = stacked @ stacked.T
    gram = [[float(value) for value in row] for row in gram_tensor]
    norms = torch.linalg.vector_norm(stacked, dim=1)
    cosine: list[list[float | None]] = []
    for row_index in range(len(vectors)):
        row: list[float | None] = []
        for column_index in range(len(vectors)):
            denominator = float(norms[row_index] * norms[column_index])
            row.append(
                None
                if denominator == 0.0
                else float(gram_tensor[row_index, column_index] / denominator)
            )
        cosine.append(row)
    return gram, cosine


def _add_vocab_projections(
    report: dict[str, Any],
    effects: dict[str, torch.Tensor],
    manifest: dict[str, Any],
    *,
    top_k: int,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(manifest["model_id"])
    revision = manifest.get("model_revision")
    tokenizer_revision = manifest.get("tokenizer_revision") or revision
    if not revision or not tokenizer_revision:
        raise ValueError("vocabulary projection requires immutable model/tokenizer revisions")
    dtype_name = str(manifest.get("dtype") or "bf16")
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported manifest dtype for vocabulary projection: {dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(hf_model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != revision:
        raise ValueError(
            f"loaded model revision mismatch: expected={revision}, actual={resolved_revision}"
        )
    config_hash = sha256_json(hf_model.config.to_dict())
    if config_hash != manifest.get("model_config_sha256"):
        raise ValueError("loaded model config hash does not match lens manifest")
    model = jlens.from_hf(hf_model, tokenizer)
    if int(model.d_model) != int(manifest["d_model"]):
        raise ValueError(
            f"loaded model width mismatch: model={model.d_model}, manifest={manifest['d_model']}"
        )

    projections: dict[str, Any] = {}
    with torch.inference_mode():
        for label, effect in effects.items():
            logits = (
                model.unembed(effect.to(model.input_device).unsqueeze(0))
                .float()
                .cpu()
                .squeeze(0)
            )
            count = min(top_k, int(logits.numel()))
            positive_values, positive_ids = torch.topk(logits, count)
            negative_values, negative_ids = torch.topk(-logits, count)

            def rows(
                ids: torch.Tensor, values: torch.Tensor, *, negate: bool
            ) -> list[dict[str, Any]]:
                result: list[dict[str, Any]] = []
                for token_id, value in zip(ids.tolist(), values.tolist(), strict=True):
                    result.append(
                        {
                            "token_id": int(token_id),
                            "token_text": tokenizer.decode([int(token_id)]),
                            "logit": float(-value if negate else value),
                        }
                    )
                return result

            projections[label] = {
                "top_positive": rows(positive_ids, positive_values, negate=False),
                "top_negative": rows(negative_ids, negative_values, negate=True),
            }
    report["vocabulary_projections"] = {
        "status": "computed",
        "method": "Jacobian transport, then exact final norm and LM head",
        "top_k": top_k,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "model_config_sha256": config_hash,
        "effects": projections,
    }


def build_report(
    *,
    lens_path: Path,
    manifest_path: Path,
    metadata_path: Path,
    alpha_values: list[str] | None = None,
    skip_vocab: bool = True,
    top_k: int = 20,
    require_complete: bool = False,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    lens, manifest = _load_lens(lens_path, manifest_path)
    components, vector_provenance = _load_vectors(
        metadata_path,
        manifest,
        expected_width=lens.d_model,
    )
    configured_labels = [component["label"] for component in components]
    covered = [component for component in components if component["layer"] in lens.source_layers]
    covered_labels = {component["label"] for component in covered}
    missing = [
        component["label"]
        for component in components
        if component["label"] not in covered_labels
    ]
    if not covered:
        raise ValueError("lens does not cover any configured trait/layer components")
    if require_complete and missing:
        raise ValueError(f"lens is missing configured components: {missing}")

    traits = {component["trait"] for component in components}
    alphas = _parse_alphas(alpha_values or [], traits)
    transported: dict[str, torch.Tensor] = {}
    component_rows: list[dict[str, Any]] = []
    for component in covered:
        label = component["label"]
        layer = component["layer"]
        vector = component["vector"].float()
        jacobian = lens.jacobians[layer].float()
        effect = lens.transport(vector, layer).float().cpu()
        raw_norm = float(torch.linalg.vector_norm(vector))
        transported_norm = float(torch.linalg.vector_norm(effect))
        directional_gain = transported_norm / raw_norm
        isotropic_rms_gain = float(torch.linalg.vector_norm(jacobian)) / math.sqrt(lens.d_model)
        gain_ratio = (
            directional_gain / isotropic_rms_gain if isotropic_rms_gain != 0.0 else None
        )
        transported[label] = effect
        component_rows.append(
            {
                "label": label,
                "trait": component["trait"],
                "trait_name": component["trait_name"],
                "layer": layer,
                "vector_store_id": component["vector_store_id"],
                "polarity": component["polarity"],
                "vector_sha256": component["vector_sha256"],
                "train_set_hash": component["index_entry"].get("train_set_hash"),
                "eval_set_hash": component["index_entry"].get("eval_set_hash"),
                "raw_norm": raw_norm,
                "transported_norm": transported_norm,
                "directional_gain": directional_gain,
                "isotropic_rms_gain": isotropic_rms_gain,
                "gain_ratio": gain_ratio,
            }
        )

    labels = [row["label"] for row in component_rows]
    component_vectors = [transported[label] for label in labels]
    component_gram, component_cosine = _matrix(component_vectors)
    component_pairs = []
    for left_index, right_index in combinations(range(len(labels)), 2):
        component_pairs.append(
            {
                "left": labels[left_index],
                "right": labels[right_index],
                "same_trait": component_rows[left_index]["trait"]
                == component_rows[right_index]["trait"],
                "dot": component_gram[left_index][right_index],
                "cosine": component_cosine[left_index][right_index],
            }
        )
    component_pairs.sort(
        key=lambda pair: abs(pair["cosine"]) if pair["cosine"] is not None else -1.0,
        reverse=True,
    )

    by_trait: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_trait[row["trait"]].append(row)
    trait_effects: dict[str, torch.Tensor] = {}
    trait_rows: list[dict[str, Any]] = []
    for trait in sorted(by_trait):
        rows = by_trait[trait]
        effect = torch.stack([transported[row["label"]] for row in rows]).sum(dim=0)
        trait_effects[trait] = effect
        component_norm_sum = sum(row["transported_norm"] for row in rows)
        effect_norm = float(torch.linalg.vector_norm(effect))
        within_pairs = [
            pair["cosine"]
            for pair in component_pairs
            if pair["same_trait"]
            and pair["left"].split("@", 1)[0] == trait
            and pair["cosine"] is not None
        ]
        trait_rows.append(
            {
                "trait": trait,
                "trait_name": rows[0]["trait_name"],
                "layers": [row["layer"] for row in rows],
                "component_count": len(rows),
                "alpha": alphas[trait],
                "transported_sum_norm": effect_norm,
                "sum_component_norms": component_norm_sum,
                "coherence_ratio": effect_norm / component_norm_sum
                if component_norm_sum != 0.0
                else None,
                "mean_within_trait_cosine": sum(within_pairs) / len(within_pairs)
                if within_pairs
                else None,
                "mean_directional_gain": sum(row["directional_gain"] for row in rows)
                / len(rows),
                "mean_gain_ratio": sum(
                    row["gain_ratio"] for row in rows if row["gain_ratio"] is not None
                )
                / len([row for row in rows if row["gain_ratio"] is not None])
                if any(row["gain_ratio"] is not None for row in rows)
                else None,
                "alpha_weighted_norm": abs(alphas[trait]) * effect_norm,
            }
        )

    trait_labels = sorted(trait_effects)
    trait_gram, trait_cosine = _matrix([trait_effects[trait] for trait in trait_labels])
    trait_pairs = []
    for left_index, right_index in combinations(range(len(trait_labels)), 2):
        trait_pairs.append(
            {
                "left": trait_labels[left_index],
                "right": trait_labels[right_index],
                "dot": trait_gram[left_index][right_index],
                "cosine": trait_cosine[left_index][right_index],
            }
        )
    trait_pairs.sort(
        key=lambda pair: abs(pair["cosine"]) if pair["cosine"] is not None else -1.0,
        reverse=True,
    )

    weighted_effects = {
        trait: effect * float(alphas[trait]) for trait, effect in trait_effects.items()
    }
    combined = torch.stack(list(weighted_effects.values())).sum(dim=0)
    combined_norm = float(torch.linalg.vector_norm(combined))
    contributions = []
    for trait in trait_labels:
        contribution = weighted_effects[trait]
        contributions.append(
            {
                "trait": trait,
                "alpha": alphas[trait],
                "norm": float(torch.linalg.vector_norm(contribution)),
                "cosine_with_combined": _cosine(contribution, combined),
                "dot_with_combined": float(torch.dot(contribution, combined)),
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "lens_path": str(lens_path.resolve()),
            "lens_sha256": sha256_file(lens_path),
            "lens_manifest_path": str(manifest_path.resolve()),
            "lens_manifest_sha256": sha256_file(manifest_path),
            "lens_id": manifest.get("lens_id"),
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "tokenizer_revision": manifest.get("tokenizer_revision"),
            "model_config_sha256": manifest.get("model_config_sha256"),
            "jlens_git_commit": manifest.get("jlens_git_commit"),
            "jlens_version": manifest.get("jlens_version"),
            "d_model": lens.d_model,
            "n_layers": manifest.get("n_layers"),
            "target_layer": manifest.get("target_layer"),
            "lens_prompts": lens.n_prompts,
            **vector_provenance,
        },
        "coverage": {
            "configured_components": configured_labels,
            "lens_source_layers": list(lens.source_layers),
            "covered_components": labels,
            "missing_components": missing,
            "unconfigured_lens_layers": sorted(
                set(lens.source_layers) - {component["layer"] for component in components}
            ),
            "complete": not missing,
        },
        "alphas": alphas,
        "components": component_rows,
        "component_space": {
            "basis": f"target residual layer {manifest.get('target_layer')}",
            "labels": labels,
            "gram": component_gram,
            "cosine": component_cosine,
            "pairs_by_absolute_cosine": component_pairs,
        },
        "traits": trait_rows,
        "trait_space": {
            "aggregation": "sum of first-order transported layer components per trait",
            "labels": trait_labels,
            "gram": trait_gram,
            "cosine": trait_cosine,
            "pairs_by_absolute_cosine": trait_pairs,
        },
        "alpha_weighted_combined_effect": {
            "norm": combined_norm,
            "sum_contribution_norms": sum(item["norm"] for item in contributions),
            "coherence_ratio": combined_norm / sum(item["norm"] for item in contributions)
            if sum(item["norm"] for item in contributions) != 0.0
            else None,
            "contributions": contributions,
        },
        "vocabulary_projections": {
            "status": "skipped",
            "reason": "--skip-vocab requested" if skip_vocab else None,
        },
    }
    projection_effects = {
        **{f"component:{label}": vector for label, vector in transported.items()},
        **{f"trait:{trait}": vector for trait, vector in trait_effects.items()},
        "combined:alpha_weighted": combined,
    }
    if not skip_vocab:
        _add_vocab_projections(report, projection_effects, manifest, top_k=top_k)
    report["analysis_sha256"] = sha256_json(
        {
            key: value
            for key, value in report.items()
            if key not in {"created_at", "analysis_sha256"}
        }
    )
    return report


def _format_number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    provenance = report["provenance"]
    coverage = report["coverage"]
    lines = [
        "# Jacobian-space trait report",
        "",
        "This report compares steering components only after Jacobian transport into the "
        "common target-layer residual basis. Cross-layer raw-vector cosines are not used.",
        "",
        "## Provenance",
        "",
        *_markdown_table(
            ["Field", "Value"],
            [
                ["Model", f"{provenance['model_id']}@{provenance['model_revision']}"],
                ["Lens", str(provenance.get("lens_id"))],
                ["Lens SHA-256", provenance["lens_sha256"]],
                ["Vector metadata SHA-256", provenance["vector_metadata_sha256"]],
                ["Vector index SHA-256", provenance["vector_index_sha256"]],
                ["Target layer", str(provenance["target_layer"])],
                ["Width", str(provenance["d_model"])],
                ["Fit prompts", str(provenance["lens_prompts"])],
                ["Analysis SHA-256", report["analysis_sha256"]],
            ],
        ),
        "",
        "## Coverage",
        "",
        f"Covered **{len(coverage['covered_components'])} / "
        f"{len(coverage['configured_components'])}** configured components: "
        f"{', '.join(coverage['covered_components'])}.",
    ]
    if coverage["missing_components"]:
        lines.extend(
            ["", f"Missing from this lens: {', '.join(coverage['missing_components'])}."]
        )

    lines.extend(
        [
            "",
            "## Component transport",
            "",
            *_markdown_table(
                [
                    "Component",
                    "Raw norm",
                    "Transported norm",
                    "Directional gain",
                    "Isotropic RMS gain",
                    "Gain ratio",
                ],
                [
                    [
                        row["label"],
                        _format_number(row["raw_norm"]),
                        _format_number(row["transported_norm"]),
                        _format_number(row["directional_gain"]),
                        _format_number(row["isotropic_rms_gain"]),
                        _format_number(row["gain_ratio"]),
                    ]
                    for row in report["components"]
                ],
            ),
            "",
            "Directional gain is `||Jv|| / ||v||`. Isotropic RMS gain is "
            "`||J||_F / sqrt(d)`. Their ratio shows whether a steering direction is "
            "amplified more or less than an isotropic unit direction at that layer.",
            "",
            "## Transported component cosine matrix",
            "",
        ]
    )
    component_space = report["component_space"]
    lines.extend(
        _markdown_table(
            ["Component", *component_space["labels"]],
            [
                [label, *[_format_number(value) for value in row]]
                for label, row in zip(
                    component_space["labels"], component_space["cosine"], strict=True
                )
            ],
        )
    )
    lines.extend(["", "## Trait aggregates", ""])
    lines.extend(
        _markdown_table(
            [
                "Trait",
                "Layers",
                "Alpha",
                "Aggregate norm",
                "Coherence",
                "Mean within-trait cosine",
            ],
            [
                [
                    row["trait"],
                    ", ".join(str(layer) for layer in row["layers"]),
                    _format_number(row["alpha"]),
                    _format_number(row["transported_sum_norm"]),
                    _format_number(row["coherence_ratio"]),
                    _format_number(row["mean_within_trait_cosine"]),
                ]
                for row in report["traits"]
            ],
        )
    )
    trait_space = report["trait_space"]
    lines.extend(["", "### Trait aggregate cosine matrix", ""])
    lines.extend(
        _markdown_table(
            ["Trait", *trait_space["labels"]],
            [
                [label, *[_format_number(value) for value in row]]
                for label, row in zip(trait_space["labels"], trait_space["cosine"], strict=True)
            ],
        )
    )
    combined = report["alpha_weighted_combined_effect"]
    lines.extend(
        [
            "",
            "## Alpha-weighted combined effect",
            "",
            f"Combined norm: **{_format_number(combined['norm'])}**; coherence ratio: "
            f"**{_format_number(combined['coherence_ratio'])}**.",
            "",
            *_markdown_table(
                ["Trait", "Alpha", "Contribution norm", "Cosine with combined"],
                [
                    [
                        row["trait"],
                        _format_number(row["alpha"]),
                        _format_number(row["norm"]),
                        _format_number(row["cosine_with_combined"]),
                    ]
                    for row in combined["contributions"]
                ],
            ),
            "",
            "## Strongest component interactions",
            "",
        ]
    )
    for pair in component_space["pairs_by_absolute_cosine"][:10]:
        lines.append(
            f"- {pair['left']} ↔ {pair['right']}: cosine "
            f"{_format_number(pair['cosine'])}"
        )

    vocabulary = report["vocabulary_projections"]
    lines.extend(["", "## Vocabulary projections", ""])
    if vocabulary["status"] == "skipped":
        lines.append("Skipped for CPU/offline analysis.")
    else:
        lines.append(
            "Each direction was passed through the exact model's final norm and LM head. "
            "Positive and negative lists are directional logit projections, not generated text."
        )
        for label, projection in vocabulary["effects"].items():
            positive = ", ".join(
                f"`{row['token_text'].replace('`', '')}` ({_format_number(row['logit'])})"
                for row in projection["top_positive"]
            )
            negative = ", ".join(
                f"`{row['token_text'].replace('`', '')}` ({_format_number(row['logit'])})"
                for row in projection["top_negative"]
            )
            lines.extend(
                ["", f"### {label}", "", f"Positive: {positive}", "", f"Negative: {negative}"]
            )
    return "\n".join(lines) + "\n"


def _write_pair_atomic(json_path: Path, markdown_path: Path, report: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    markdown_text = render_markdown(report)
    json_temp = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    markdown_temp = markdown_path.with_name(f".{markdown_path.name}.tmp.{os.getpid()}")
    try:
        json_temp.write_text(json_text, encoding="utf-8")
        markdown_temp.write_text(markdown_text, encoding="utf-8")
        json_temp.replace(json_path)
        markdown_temp.replace(markdown_path)
    finally:
        json_temp.unlink(missing_ok=True)
        markdown_temp.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    if args.output_prefix.suffix.lower() in {".json", ".md"}:
        raise ValueError("--output-prefix must not include .json or .md")
    report = build_report(
        lens_path=args.lens,
        manifest_path=args.lens_manifest,
        metadata_path=args.vector_metadata,
        alpha_values=args.alpha,
        skip_vocab=args.skip_vocab,
        top_k=args.top_k,
        require_complete=args.require_complete,
    )
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    _write_pair_atomic(json_path, markdown_path, report)
    print(
        json.dumps(
            {
                "analysis_sha256": report["analysis_sha256"],
                "complete": report["coverage"]["complete"],
                "components": len(report["components"]),
                "json": str(json_path),
                "json_sha256": sha256_file(json_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

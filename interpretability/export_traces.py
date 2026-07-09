from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jlens
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steering.hooks import SteeringController  # noqa: E402

try:  # Supports both `python -m` and direct script execution.
    from .common import (
        read_inference_events,
        resolve_event_paths,
        sha256_file,
        sha256_json,
        write_json_atomic,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        read_inference_events,
        resolve_event_paths,
        sha256_file,
        sha256_json,
        write_json_atomic,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay simulator inference events and export Jacobian Lens top-k traces"
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--lens-manifest", type=Path, required=True)
    parser.add_argument("--vector-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trace-id", action="append", default=[])
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--allow-unpinned-event", action="store_true")
    parser.add_argument("--replay-atol", type=float, default=5e-3)
    return parser.parse_args()


def _load_manifest(path: Path, lens_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(lens_path)
    if actual_hash != manifest.get("lens_sha256"):
        raise ValueError(
            f"lens hash mismatch: manifest={manifest.get('lens_sha256')}, actual={actual_hash}"
        )
    return manifest


def _uniform_value(events: list[dict[str, Any]], field: str) -> Any:
    values = {json.dumps(event.get(field), sort_keys=True) for event in events}
    if len(values) != 1:
        raise ValueError(f"events disagree on {field}: {sorted(values)}")
    return events[0].get(field)


def _validate_events(
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    allow_unpinned: bool,
) -> None:
    if not events:
        raise ValueError("no inference events selected")
    model_id = _uniform_value(events, "model_id")
    model_revision = _uniform_value(events, "model_revision")
    _uniform_value(events, "tokenizer_revision")
    _uniform_value(events, "quantization")
    if model_id != manifest.get("model_id"):
        raise ValueError(
            f"event/lens model mismatch: event={model_id!r}, lens={manifest.get('model_id')!r}"
        )
    if not model_revision and not allow_unpinned:
        raise ValueError(
            "inference event has no immutable model revision; capture a new event or pass "
            "--allow-unpinned-event for exploratory use"
        )
    lens_revision = manifest.get("model_revision")
    if model_revision and lens_revision and model_revision != lens_revision:
        raise ValueError(
            f"event/lens revision mismatch: event={model_revision!r}, lens={lens_revision!r}"
        )
    for event in events:
        input_ids = event.get("input_ids") or []
        attention_mask = event.get("attention_mask") or []
        generated_ids = event.get("generated_ids") or []
        if not input_ids or len(input_ids) != len(attention_mask):
            raise ValueError(f"trace {event.get('trace_id')} has invalid prompt token arrays")
        if not generated_ids:
            raise ValueError(f"trace {event.get('trace_id')} has no generated token IDs")


def _resolve_vector_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    candidates = [path, root / path.name] if path.is_absolute() else [root / path, root / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"steering vector not found: {recorded}; root={root}")


def _vector_entries(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    index_path = root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing steering vector index: {index_path}")
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            entries[(str(payload["vector_store_id"]), int(payload["layer_id"]))] = payload
    return entries


def _load_trait_vectors(
    event: dict[str, Any],
    metadata_path: Path,
    *,
    expected_width: int,
) -> dict[str, dict[int, torch.Tensor]]:
    alphas = event.get("effective_alphas") or {}
    if not any(float(value) != 0.0 for value in alphas.values()):
        return {}
    config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    root = (metadata_path.parent / config["vector_root"]).resolve()
    entries = _vector_entries(root)
    trait_config = config.get("traits") or {}
    vector_ids = event.get("steering_vector_ids") or {}
    expected_hashes = event.get("steering_vector_hashes") or {}
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for trait, alpha in alphas.items():
        if float(alpha) == 0.0:
            continue
        vector_id = vector_ids.get(trait)
        hashes = expected_hashes.get(trait) or {}
        if not vector_id or not hashes:
            raise ValueError(f"trace lacks vector identity/hash data for active trait {trait}")
        polarity = float((trait_config.get(trait) or {}).get("polarity", 1.0))
        if polarity not in (-1.0, 1.0):
            raise ValueError(f"invalid polarity for trait {trait}: {polarity}")
        by_layer: dict[int, torch.Tensor] = {}
        for layer_text, expected_hash in hashes.items():
            layer = int(layer_text)
            entry = entries.get((str(vector_id), layer))
            if entry is None:
                raise ValueError(f"vector index lacks {vector_id} layer {layer}")
            vector_path = _resolve_vector_path(root, str(entry["vector_path"]))
            actual_hash = sha256_file(vector_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"vector hash mismatch for {trait} layer {layer}: "
                    f"event={expected_hash}, actual={actual_hash}"
                )
            array = np.load(vector_path, allow_pickle=False)
            if array.ndim != 1 or int(array.shape[0]) != expected_width:
                raise ValueError(
                    f"vector shape mismatch for {trait} layer {layer}: {array.shape}, "
                    f"expected ({expected_width},)"
                )
            by_layer[layer] = torch.from_numpy(array.astype(np.float32) * polarity)
        vectors[trait] = by_layer
    return vectors


def _load_model(event: dict[str, Any]):
    model_id = str(event["model_id"])
    revision = event.get("model_revision")
    tokenizer_revision = event.get("tokenizer_revision") or revision
    quantization = event.get("quantization")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        dtype_name = event.get("inference_dtype") or "bfloat16"
        dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(dtype_name))
        if dtype is None:
            raise ValueError(f"unsupported inference dtype: {dtype_name!r}")
        kwargs["dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), tokenizer


def _rank(logits: torch.Tensor, token_id: int) -> int:
    score = logits[token_id]
    return int((logits > score).sum().item()) + 1


@torch.no_grad()
def _trace_condition(
    *,
    hf_model,
    tokenizer,
    model,
    lens,
    event: dict[str, Any],
    trait_vectors: dict[str, dict[int, torch.Tensor]],
    condition: str,
    top_k: int,
    replay_atol: float,
) -> list[dict[str, Any]]:
    prompt_ids = [int(token) for token in event["input_ids"]]
    generated_ids = [int(token) for token in event["generated_ids"]]
    full_ids = prompt_ids + generated_ids
    attention_mask = [int(value) for value in event["attention_mask"]] + [1] * len(generated_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.input_device)
    mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=model.input_device)
    prompt_mask = torch.tensor(
        [True] * len(prompt_ids) + [False] * len(generated_ids), dtype=torch.bool
    )
    alphas = {
        trait: (float(value) if condition == "observed" else 0.0)
        for trait, value in (event.get("effective_alphas") or {}).items()
    }

    controller = None
    if trait_vectors:
        controller = SteeringController(hf_model, trait_vectors)
        controller.register()
        controller.set_alphas(alphas, prompt_mask=prompt_mask)

    record_layers = sorted(set(lens.source_layers) | {model.n_layers - 1})
    try:
        with jlens.ActivationRecorder(model.layers, at=record_layers) as recorder:
            output = hf_model(input_ids=input_ids, attention_mask=mask_tensor, use_cache=False)
        activations = {layer: recorder.activations[layer].detach() for layer in record_layers}
    finally:
        if controller:
            controller.remove()

    final_logits = model.unembed(activations[model.n_layers - 1][0].float()).float()
    direct_logits = output.logits[0].float().to(final_logits.device)
    replay_error = float((final_logits - direct_logits).abs().max().item())
    if replay_error > replay_atol:
        raise ValueError(
            f"trace {event['trace_id']} {condition} final-logit replay error "
            f"{replay_error:.6g} exceeds atol={replay_atol}"
        )

    source_positions = range(len(prompt_ids) - 1, len(full_ids) - 1)
    rows: list[dict[str, Any]] = []
    for layer in lens.source_layers:
        residuals = activations[layer][0, list(source_positions), :].float()
        transported = lens.transport(residuals, layer)
        layer_logits = model.unembed(transported).float().cpu()
        for row_index, source_position in enumerate(source_positions):
            predicted_position = source_position + 1
            actual_token_id = full_ids[predicted_position]
            logits = layer_logits[row_index]
            values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
            actual_rank = _rank(logits, actual_token_id)
            for rank_index, (token_id, logit) in enumerate(
                zip(indices.tolist(), values.tolist(), strict=True), 1
            ):
                rows.append(
                    {
                        "trace_id": event["trace_id"],
                        "run_id": event["run_id"],
                        "tick": int(event["tick"]),
                        "agent_id": event["agent_id"],
                        "action_id": event["action_id"],
                        "selected_action_type": event.get("selected_action_type"),
                        "decision_source": event.get("decision_source"),
                        "condition": condition,
                        "position_phase": (
                            "prompt" if source_position < len(prompt_ids) else "generated"
                        ),
                        "source_position": source_position,
                        "predicted_position": predicted_position,
                        "generated_offset": predicted_position - len(prompt_ids),
                        "layer": layer,
                        "rank": rank_index,
                        "lens_token_id": token_id,
                        "lens_token_text": tokenizer.decode([token_id]),
                        "lens_logit": float(logit),
                        "actual_next_token_id": actual_token_id,
                        "actual_next_token_text": tokenizer.decode([actual_token_id]),
                        "actual_next_token_rank": actual_rank,
                        "is_actual_next_token": token_id == actual_token_id,
                        "final_logit_replay_max_abs_error": replay_error,
                    }
                )
    return rows


def main() -> None:
    args = _parse_args()
    event_paths = resolve_event_paths(args.events)
    events = read_inference_events(event_paths)
    if args.trace_id:
        selected = set(args.trace_id)
        events = [event for event in events if event.get("trace_id") in selected]
    if args.limit is not None:
        events = events[: args.limit]
    manifest = _load_manifest(args.lens_manifest, args.lens)
    _validate_events(events, manifest, allow_unpinned=args.allow_unpinned_event)

    hf_model, tokenizer = _load_model(events[0])
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(str(args.lens))
    if lens.d_model != model.d_model or manifest.get("d_model") != model.d_model:
        raise ValueError(
            f"lens/model width mismatch: lens={lens.d_model}, model={model.d_model}, "
            f"manifest={manifest.get('d_model')}"
        )
    config_hash = sha256_json(hf_model.config.to_dict())
    if manifest.get("model_config_sha256") != config_hash:
        raise ValueError("loaded model config hash does not match lens manifest")

    rows: list[dict[str, Any]] = []
    for event in events:
        trait_vectors = _load_trait_vectors(
            event,
            args.vector_metadata,
            expected_width=model.d_model,
        )
        conditions = ["observed", "neutral_replay"] if args.include_neutral else ["observed"]
        for condition in conditions:
            rows.extend(
                _trace_condition(
                    hf_model=hf_model,
                    tokenizer=tokenizer,
                    model=model,
                    lens=lens,
                    event=event,
                    trait_vectors=trait_vectors,
                    condition=condition,
                    top_k=args.top_k,
                    replay_atol=args.replay_atol,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    trace_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "events": len(events),
        "trace_ids": [event["trace_id"] for event in events],
        "conditions": sorted({row["condition"] for row in rows}),
        "rows": len(rows),
        "top_k": args.top_k,
        "lens_id": manifest["lens_id"],
        "lens_sha256": manifest["lens_sha256"],
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "input_event_files": [str(path) for path in event_paths],
    }
    write_json_atomic(args.output.with_suffix(".manifest.json"), trace_manifest)
    print(json.dumps(trace_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

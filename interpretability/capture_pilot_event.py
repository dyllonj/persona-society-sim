from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steering.hooks import SteeringController  # noqa: E402

try:
    from .common import sha256_file, sha256_json, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from common import sha256_file, sha256_json, write_json_atomic  # type: ignore


DEFAULT_PROMPT = """System: You are agent-pilot, an activation-steered research agent.
Return ONLY one JSON object with keys action, params, and utterance.
Supported actions: talk, work, research, cite, submit_report.
Current goal: coordinate a careful, reproducible analysis of a newly collected result.
Observation: A collaborator asks which evidence to check before reporting a conclusion.
JSON:"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one exact, steered agent inference for Jacobian Lens validation"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--vector-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="E=0.8,A=0.5,C=0.6")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _parse_alphas(value: str) -> dict[str, float]:
    alphas: dict[str, float] = {}
    for item in value.split(","):
        trait, separator, raw_alpha = item.strip().partition("=")
        if not separator or not trait:
            raise ValueError(f"invalid alpha entry {item!r}; expected TRAIT=value")
        alphas[trait] = float(raw_alpha)
    if not alphas or not any(alpha != 0.0 for alpha in alphas.values()):
        raise ValueError("at least one nonzero alpha is required")
    return alphas


def _resolve_vector_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    candidates = [path, root / path.name] if path.is_absolute() else [root / path, root / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"steering vector not found: {recorded}; root={root}")


def _load_vectors(
    metadata_path: Path,
    alphas: dict[str, float],
) -> tuple[
    dict[str, dict[int, torch.Tensor]],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    root = (metadata_path.parent / config["vector_root"]).resolve()
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[(str(entry["vector_store_id"]), int(entry["layer_id"]))] = entry

    vectors: dict[str, dict[int, torch.Tensor]] = {}
    vector_ids: dict[str, str] = {}
    vector_hashes: dict[str, dict[str, str]] = {}
    for trait, alpha in alphas.items():
        if alpha == 0.0:
            continue
        trait_config = (config.get("traits") or {}).get(trait)
        if not trait_config:
            raise ValueError(f"no vector metadata for active trait {trait}")
        vector_id = str(trait_config["vector_store_id"])
        polarity = float(trait_config.get("polarity", 1.0))
        if polarity not in (-1.0, 1.0):
            raise ValueError(f"invalid polarity for trait {trait}: {polarity}")
        by_layer: dict[int, torch.Tensor] = {}
        hashes: dict[str, str] = {}
        for raw_layer in trait_config["layers"]:
            layer = int(raw_layer)
            entry = entries.get((vector_id, layer))
            if entry is None:
                raise ValueError(f"vector index lacks {vector_id} layer {layer}")
            vector_path = _resolve_vector_path(root, str(entry["vector_path"]))
            array = np.load(vector_path, allow_pickle=False)
            if array.ndim != 1:
                raise ValueError(f"vector {vector_path} must be one-dimensional")
            by_layer[layer] = torch.from_numpy(array.astype(np.float32) * polarity)
            hashes[str(layer)] = sha256_file(vector_path)
        vectors[trait] = by_layer
        vector_ids[trait] = vector_id
        vector_hashes[trait] = hashes
    return vectors, vector_ids, vector_hashes


def _selected_action(raw_completion: str) -> tuple[str, str, str | None]:
    try:
        payload = json.loads(raw_completion.strip())
        action = payload.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError("JSON action is missing")
        return action, "model", None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return "talk", "planner_fallback", str(exc)


def main() -> None:
    args = _parse_args()
    alphas = _parse_alphas(args.alphas)
    vectors, vector_ids, vector_hashes = _load_vectors(args.vector_metadata, alphas)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer_revision = args.tokenizer_revision or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=tokenizer_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None) or args.model_revision
    resolved_tokenizer_revision = (
        tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision
    )
    tokens = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    controller = SteeringController(model, vectors)
    controller.register()

    def smoke_forward() -> object:
        with torch.inference_mode():
            return model(**tokens, use_cache=False)

    try:
        runtime_deltas = controller.measure_runtime_deltas(smoke_forward)
        controller.set_alphas(
            alphas,
            prompt_mask=torch.ones_like(tokens["input_ids"][0], dtype=torch.bool),
        )
        with torch.inference_mode():
            output = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            )
    finally:
        controller.clear_prompt_metadata()
        controller.remove()

    input_ids = tokens["input_ids"][0].detach().cpu().tolist()
    attention_mask = tokens["attention_mask"][0].detach().cpu().tolist()
    generated_ids = output[0, len(input_ids) :].detach().cpu().tolist()
    raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    selected_action, decision_source, parse_error = _selected_action(raw_completion)
    prompt_hash = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    trace_id = f"pilot-{sha256_json([prompt_hash, alphas, resolved_model_revision])[:16]}"
    event = {
        "trace_id": trace_id,
        "schema_version": "1.0",
        "run_id": "jacobian-lens-pilot",
        "tick": 0,
        "agent_id": "agent-pilot",
        "action_id": "pilot-action-0",
        "selected_action_type": selected_action,
        "decision_source": decision_source,
        "decision_parse_error": parse_error,
        "cognitive_phase": "action_generation",
        "capture_reason": "gpu_pilot",
        "prompt_hash": prompt_hash,
        "prompt_text": args.prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "generated_ids": generated_ids,
        "prompt_token_count": len(input_ids),
        "generated_token_count": len(generated_ids),
        "raw_completion": raw_completion,
        "model_id": args.model_id,
        "model_revision": resolved_model_revision,
        "tokenizer_revision": resolved_tokenizer_revision,
        "inference_dtype": str(dtype).removeprefix("torch."),
        "quantization": None,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "sampling_seed": None,
        "effective_alphas": alphas,
        "steering_applied": True,
        "steering_vector_ids": vector_ids,
        "steering_vector_hashes": vector_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "trace_id": trace_id,
        "event_sha256": sha256_file(args.output),
        "steering_runtime_delta_norms": runtime_deltas,
        "selected_action_type": selected_action,
        "decision_source": decision_source,
        "generated_token_count": len(generated_ids),
    }
    write_json_atomic(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({**manifest, "raw_completion": raw_completion}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

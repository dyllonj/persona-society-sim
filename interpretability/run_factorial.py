"""Run paired, live generations across personality-steering conditions.

The runner deliberately keeps prompts and sampling seeds paired while allowing
each condition to follow its own generated-token path. It loads the model once,
registers one controller containing both observed and placebo vectors, and
writes replay-complete JSONL records plus a content-hashed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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
    from .common import load_prompts, sha256_file, sha256_json, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        load_prompts,
        sha256_file,
        sha256_json,
        write_json_atomic,
    )


TRAITS = ("E", "A", "C")
CONDITION_ORDER = ("neutral", "E_only", "A_only", "C_only", "E_A_C", "placebo_shuffled")
PLACEBO_ALGORITHM = "numpy-pcg64-coordinate-permutation-v1"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
PERSONA_LABELS = re.compile(
    r"\b(?:"
    r"persona|personality|personality\s+trait|big\s+five|"
    r"extraversion|extravert(?:ed)?|extroversion|extrovert(?:ed)?|introvert(?:ed)?|"
    r"agreeableness|agreeable|disagreeable|"
    r"conscientiousness|conscientious|unconscientious"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Condition:
    """A logical experimental arm and its controller-level alpha map."""

    name: str
    effective_alphas: dict[str, float]
    controller_alphas: dict[str, float]
    vector_mode: str


@dataclass(frozen=True)
class VectorBundle:
    """Loaded vectors and immutable provenance needed for event records."""

    vectors: dict[str, dict[int, torch.Tensor]]
    vector_ids: dict[str, str]
    source_hashes: dict[str, dict[str, str]]
    applied_hashes: dict[str, dict[str, str]]
    polarities: dict[str, float]
    mapping: dict[str, dict[str, Any]]
    metadata_sha256: str
    index_sha256: str


@dataclass(frozen=True)
class FactorialPrompt:
    """One condition-blind prompt plus analysis-only stratum provenance."""

    prompt_id: str
    text: str
    origin_stratum: str | None = None
    source_id: str | None = None
    source_file: str | None = None
    source_record_index: int | None = None


def parse_base_alphas(value: str) -> dict[str, float]:
    """Parse exactly one finite, nonzero magnitude for each experimental trait."""

    parsed: dict[str, float] = {}
    for item in value.split(","):
        trait, separator, raw = item.strip().partition("=")
        if not separator or trait not in TRAITS:
            raise ValueError(f"invalid alpha entry {item!r}; expected E=value,A=value,C=value")
        if trait in parsed:
            raise ValueError(f"duplicate alpha for trait {trait}")
        alpha = float(raw)
        if not np.isfinite(alpha) or alpha == 0.0:
            raise ValueError(f"alpha for trait {trait} must be finite and nonzero")
        parsed[trait] = alpha
    missing = set(TRAITS) - parsed.keys()
    if missing:
        raise ValueError(f"missing base alphas for traits: {sorted(missing)}")
    return {trait: parsed[trait] for trait in TRAITS}


def build_conditions(base_alphas: Mapping[str, float]) -> tuple[Condition, ...]:
    """Construct the six prespecified arms without touching model state."""

    base = {trait: float(base_alphas[trait]) for trait in TRAITS}
    zero = {trait: 0.0 for trait in TRAITS}
    controller_zero = {
        controller_trait: 0.0
        for trait in TRAITS
        for controller_trait in (trait, f"placebo_{trait}")
    }
    conditions: list[Condition] = [
        Condition("neutral", dict(zero), dict(controller_zero), "none"),
    ]
    for active in TRAITS:
        effective = {trait: (base[trait] if trait == active else 0.0) for trait in TRAITS}
        controller_alphas = dict(controller_zero)
        controller_alphas.update(effective)
        conditions.append(Condition(f"{active}_only", effective, controller_alphas, "identity"))
    combined_controller_alphas = dict(controller_zero)
    combined_controller_alphas.update(base)
    conditions.append(
        Condition("E_A_C", dict(base), combined_controller_alphas, "identity")
    )
    placebo_controller_alphas = dict(controller_zero)
    placebo_controller_alphas.update(
        {f"placebo_{trait}": base[trait] for trait in TRAITS}
    )
    conditions.append(
        Condition(
            "placebo_shuffled",
            dict(base),
            placebo_controller_alphas,
            "coordinate_permutation",
        )
    )
    result = tuple(conditions)
    if tuple(condition.name for condition in result) != CONDITION_ORDER:
        raise AssertionError("factorial condition order changed")
    return result


def prompt_seed(base_seed: int, prompt_index: int) -> int:
    """Derive a stable, paired seed in PyTorch's signed 63-bit range."""

    if prompt_index < 0:
        raise ValueError("prompt_index must be non-negative")
    digest = hashlib.sha256(f"{int(base_seed)}:{prompt_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def placebo_permutation_seed(
    placebo_seed: int,
    trait: str,
    layer: int,
    source_hash: str,
) -> int:
    """Derive an independently reproducible seed for one vector permutation."""

    payload = f"{PLACEBO_ALGORITHM}:{int(placebo_seed)}:{trait}:{int(layer)}:{source_hash}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def shuffled_vector(vector: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return a coordinate-permuted copy, preserving the vector's exact norm."""

    if vector.ndim != 1:
        raise ValueError("placebo vectors must be one-dimensional")
    permutation = np.random.default_rng(seed).permutation(int(vector.shape[0]))
    indices = torch.from_numpy(permutation.copy()).to(vector.device)
    return vector.index_select(0, indices)


def validate_neutral_prompts(prompts: Sequence[str]) -> None:
    """Reject explicit personality labels that could prompt-confound the intervention."""

    if not prompts:
        raise ValueError("at least one prompt is required")
    for index, prompt in enumerate(prompts):
        if not prompt.strip():
            raise ValueError(f"prompt {index} is empty")
        match = PERSONA_LABELS.search(prompt)
        if match:
            raise ValueError(
                f"prompt {index} contains forbidden persona label {match.group(0)!r}; "
                "factorial prompts must be condition-blind"
            )


def load_factorial_prompts(
    path: Path,
    *,
    limit: int | None = None,
) -> list[FactorialPrompt]:
    """Load prompts while preserving stable IDs and hidden analysis strata."""

    if path.suffix.lower() != ".jsonl":
        texts = load_prompts(path, limit=limit)
        return [
            FactorialPrompt(prompt_id=f"prompt-{index:04d}", text=value)
            for index, value in enumerate(texts)
        ]

    prompts: list[FactorialPrompt] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, str):
            payload = {"text": payload}
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a string or object")
        text = next(
            (
                payload[key]
                for key in ("text", "prompt", "content")
                if isinstance(payload.get(key), str)
            ),
            None,
        )
        if not text:
            raise ValueError(f"{path}:{line_number} has no text/prompt/content string")
        prompt_id = str(payload.get("prompt_id") or f"prompt-{len(prompts):04d}")
        if not prompt_id or prompt_id in seen_ids:
            raise ValueError(f"{path}:{line_number} has a duplicate or empty prompt_id")
        expected_hash = payload.get("prompt_sha256")
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_hash is not None and str(expected_hash) != actual_hash:
            raise ValueError(
                f"{path}:{line_number} prompt hash mismatch: "
                f"record={expected_hash}, actual={actual_hash}"
            )
        raw_index = payload.get("source_record_index")
        prompts.append(
            FactorialPrompt(
                prompt_id=prompt_id,
                text=text,
                origin_stratum=(
                    str(payload["origin_stratum"])
                    if payload.get("origin_stratum") is not None
                    else None
                ),
                source_id=(
                    str(payload["source_id"])
                    if payload.get("source_id") is not None
                    else None
                ),
                source_file=(
                    str(payload["source_file"])
                    if payload.get("source_file") is not None
                    else None
                ),
                source_record_index=int(raw_index) if raw_index is not None else None,
            )
        )
        seen_ids.add(prompt_id)
        if limit is not None and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"no prompts loaded from {path}")
    return prompts


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().to(torch.float32).numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _resolve_vector_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    candidates = [path, root / path.name] if path.is_absolute() else [root / path, root / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"steering vector not found: {recorded}; root={root}")


def _vector_entries(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing steering vector index: {path}")
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        key = (str(payload["vector_store_id"]), int(payload["layer_id"]))
        previous = entries.get(key)
        if previous is not None:
            stable_fields = ("trait", "method", "vector_path", "train_set_hash")
            if any(previous.get(field) != payload.get(field) for field in stable_fields):
                raise ValueError(
                    f"conflicting duplicate vector index entry {key} at line {line_number}"
                )
        entries[key] = payload
    return entries


def load_vector_bundle(
    metadata_path: Path,
    *,
    model_id: str,
    expected_width: int,
    expected_layers: int,
    placebo_seed: int,
) -> VectorBundle:
    """Load metadata-selected vectors, enforce compatibility, and build placebos."""

    metadata_path = metadata_path.resolve()
    config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    configured_model = str((config.get("defaults") or {}).get("model") or "")
    if configured_model != model_id:
        raise ValueError(
            f"vector metadata model mismatch: metadata={configured_model!r}, runtime={model_id!r}"
        )
    configured_layer_count = (config.get("defaults") or {}).get("num_hidden_layers")
    if configured_layer_count is not None and int(configured_layer_count) != expected_layers:
        raise ValueError(
            "vector metadata layer-count mismatch: "
            f"metadata={configured_layer_count}, runtime={expected_layers}"
        )

    root = (metadata_path.parent / config["vector_root"]).resolve()
    index_path = root / "index.jsonl"
    entries = _vector_entries(root)
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    vector_ids: dict[str, str] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    applied_hashes: dict[str, dict[str, str]] = {}
    polarities: dict[str, float] = {}
    mapping: dict[str, dict[str, Any]] = {}

    for trait in TRAITS:
        trait_config = (config.get("traits") or {}).get(trait)
        if not isinstance(trait_config, dict):
            raise ValueError(f"vector metadata lacks required trait {trait}")
        vector_id = str(trait_config["vector_store_id"])
        polarity = float(trait_config.get("polarity", 1.0))
        if polarity not in (-1.0, 1.0):
            raise ValueError(f"invalid polarity for trait {trait}: {polarity}")
        layers = [int(layer) for layer in trait_config.get("layers") or []]
        if not layers:
            raise ValueError(f"trait {trait} has no configured layers")

        identity_by_layer: dict[int, torch.Tensor] = {}
        placebo_by_layer: dict[int, torch.Tensor] = {}
        source_by_layer: dict[str, str] = {}
        identity_hash_by_layer: dict[str, str] = {}
        placebo_hash_by_layer: dict[str, str] = {}
        placebo_layers: dict[str, dict[str, Any]] = {}
        for layer in layers:
            if not 0 <= layer < expected_layers:
                raise ValueError(
                    f"trait {trait} layer {layer} is outside runtime range "
                    f"[0, {expected_layers - 1}]"
                )
            entry = entries.get((vector_id, layer))
            if entry is None:
                raise ValueError(f"vector index lacks {vector_id} layer {layer}")
            entry_trait = entry.get("trait")
            if entry_trait is not None and str(entry_trait) != trait:
                raise ValueError(
                    f"vector index trait mismatch for {vector_id} layer {layer}: {entry_trait!r}"
                )
            vector_path = _resolve_vector_path(root, str(entry["vector_path"]))
            array = np.load(vector_path, allow_pickle=False)
            if array.ndim != 1 or int(array.shape[0]) != expected_width:
                raise ValueError(
                    f"vector shape mismatch for {trait} layer {layer}: {array.shape}; "
                    f"expected ({expected_width},)"
                )
            identity = torch.from_numpy(array.astype(np.float32, copy=True) * polarity)
            source_hash = sha256_file(vector_path)
            permutation_seed = placebo_permutation_seed(
                placebo_seed, trait, layer, source_hash
            )
            placebo = shuffled_vector(identity, seed=permutation_seed)
            identity_by_layer[layer] = identity
            placebo_by_layer[layer] = placebo
            source_by_layer[str(layer)] = source_hash
            identity_hash_by_layer[str(layer)] = _tensor_sha256(identity)
            placebo_hash_by_layer[str(layer)] = _tensor_sha256(placebo)
            placebo_layers[str(layer)] = {
                "source_file": vector_path.name,
                "source_file_sha256": source_hash,
                "source_applied_tensor_sha256": identity_hash_by_layer[str(layer)],
                "applied_tensor_sha256": placebo_hash_by_layer[str(layer)],
                "permutation_seed": permutation_seed,
                "algorithm": PLACEBO_ALGORITHM,
            }

        vectors[trait] = identity_by_layer
        vectors[f"placebo_{trait}"] = placebo_by_layer
        vector_ids[trait] = vector_id
        source_hashes[trait] = source_by_layer
        applied_hashes[trait] = identity_hash_by_layer
        applied_hashes[f"placebo_{trait}"] = placebo_hash_by_layer
        polarities[trait] = polarity
        mapping[trait] = {
            "mode": "identity",
            "source_trait": trait,
            "vector_store_id": vector_id,
            "polarity": polarity,
            "layers": {
                layer: {
                    "source_file_sha256": source_by_layer[layer],
                    "applied_tensor_sha256": identity_hash_by_layer[layer],
                }
                for layer in source_by_layer
            },
        }
        mapping[f"placebo_{trait}"] = {
            "mode": "coordinate_permutation",
            "source_trait": trait,
            "vector_store_id": vector_id,
            "polarity": polarity,
            "layers": placebo_layers,
        }

    return VectorBundle(
        vectors=vectors,
        vector_ids=vector_ids,
        source_hashes=source_hashes,
        applied_hashes=applied_hashes,
        polarities=polarities,
        mapping=mapping,
        metadata_sha256=sha256_file(metadata_path),
        index_sha256=sha256_file(index_path),
    )


def condition_vector_provenance(
    condition: Condition,
    bundle: VectorBundle,
) -> dict[str, dict[str, Any]]:
    """Select only the mappings that were nonzero for an arm."""

    return {
        trait: bundle.mapping[trait]
        for trait, alpha in condition.controller_alphas.items()
        if alpha != 0.0
    }


@contextmanager
def seeded_rng(seed: int, *, enabled: bool) -> Iterator[None]:
    """Fork CPU/CUDA RNG state so arm order cannot change paired samples."""

    if not enabled:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six paired live-generation personality-steering conditions"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--vector-metadata",
        type=Path,
        default=ROOT / "configs" / "steering.layers.yaml",
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="E=0.8,A=0.5,C=0.6")
    parser.add_argument("--base-seed", type=int, default=1701)
    parser.add_argument("--placebo-seed", type=int, default=2909)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for field in ("model_revision", "tokenizer_revision"):
        value = getattr(args, field, None)
        if value is not None and not IMMUTABLE_REVISION.fullmatch(value):
            raise ValueError(f"--{field.replace('_', '-')} must be an immutable 40-hex commit")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.greedy:
        if not np.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("--temperature must be finite and positive")
        if not np.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            raise ValueError("--top-p must be in (0, 1]")
    if args.output.suffix.lower() != ".jsonl":
        raise ValueError("--output must end in .jsonl")


def _model_dimensions(model) -> tuple[int, int]:
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else config
    return int(text_config.hidden_size), int(text_config.num_hidden_layers)


def _resolved_revision(component: Any, fallback: str) -> str:
    if hasattr(component, "config"):
        value = getattr(component.config, "_commit_hash", None)
    else:
        value = getattr(component, "init_kwargs", {}).get("_commit_hash")
    return str(value or fallback)


def _active_vector_fields(
    condition: Condition,
    bundle: VectorBundle,
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    ids: dict[str, str] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    applied_hashes: dict[str, dict[str, str]] = {}
    for controller_trait, alpha in condition.controller_alphas.items():
        if alpha == 0.0:
            continue
        source_trait = controller_trait.removeprefix("placebo_")
        ids[controller_trait] = bundle.vector_ids[source_trait]
        source_hashes[controller_trait] = bundle.source_hashes[source_trait]
        applied_hashes[controller_trait] = bundle.applied_hashes[controller_trait]
    return ids, source_hashes, applied_hashes


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    base_alphas = parse_base_alphas(args.alphas)
    conditions = build_conditions(base_alphas)
    prompts = load_factorial_prompts(args.prompts, limit=args.limit)
    validate_neutral_prompts([prompt.text for prompt in prompts])
    tokenizer_revision = args.tokenizer_revision or args.model_revision
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    resolved_model_revision = _resolved_revision(model, args.model_revision)
    resolved_tokenizer_revision = _resolved_revision(tokenizer, tokenizer_revision)
    if resolved_model_revision != args.model_revision:
        raise ValueError(
            f"loaded model revision {resolved_model_revision!r} "
            f"!= requested {args.model_revision!r}"
        )
    if resolved_tokenizer_revision != tokenizer_revision:
        raise ValueError(
            "loaded tokenizer revision "
            f"{resolved_tokenizer_revision!r} != requested {tokenizer_revision!r}"
        )

    hidden_size, layer_count = _model_dimensions(model)
    bundle = load_vector_bundle(
        args.vector_metadata,
        model_id=args.model_id,
        expected_width=hidden_size,
        expected_layers=layer_count,
        placebo_seed=args.placebo_seed,
    )
    controller = SteeringController(model, bundle.vectors)
    controller.register()
    run_spec = {
        "schema_version": "factorial-1.0",
        "model_id": args.model_id,
        "model_revision": resolved_model_revision,
        "tokenizer_revision": resolved_tokenizer_revision,
        "model_config_sha256": sha256_json(model.config.to_dict()),
        "dtype": str(dtype).removeprefix("torch."),
        "quantization": None,
        "conditions": [
            {
                "name": condition.name,
                "effective_alphas": condition.effective_alphas,
                "controller_alphas": condition.controller_alphas,
                "vector_mode": condition.vector_mode,
            }
            for condition in conditions
        ],
        "base_seed": args.base_seed,
        "placebo_seed": args.placebo_seed,
        "do_sample": not args.greedy,
        "temperature": None if args.greedy else args.temperature,
        "top_p": None if args.greedy else args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "prompt_hashes": [
            hashlib.sha256(prompt.text.encode()).hexdigest() for prompt in prompts
        ],
        "vector_metadata_sha256": bundle.metadata_sha256,
        "vector_index_sha256": bundle.index_sha256,
        "vector_ids": bundle.vector_ids,
        "vector_source_hashes": bundle.source_hashes,
        "vector_applied_hashes": bundle.applied_hashes,
        "vector_polarities": bundle.polarities,
        "placebo_algorithm": PLACEBO_ALGORITHM,
    }
    run_id = f"factorial-{sha256_json(run_spec)[:16]}"
    rows: list[dict[str, Any]] = []
    try:
        for prompt_index, prompt_record in enumerate(prompts):
            prompt = prompt_record.text
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = tokens["input_ids"][0].detach().cpu().tolist()
            attention_mask = tokens["attention_mask"][0].detach().cpu().tolist()
            seed = prompt_seed(args.base_seed, prompt_index)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            for condition_index, condition in enumerate(conditions):
                controller.set_alphas(
                    condition.controller_alphas,
                    prompt_mask=torch.ones_like(tokens["input_ids"][0], dtype=torch.bool),
                )
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": not args.greedy,
                    "pad_token_id": (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                }
                if not args.greedy:
                    generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
                try:
                    with seeded_rng(seed, enabled=not args.greedy), torch.inference_mode():
                        output = model.generate(**tokens, **generation_kwargs)
                finally:
                    controller.clear_prompt_metadata()

                generated = output[0, len(input_ids) :].detach().cpu()
                generated_ids = generated.tolist()
                raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
                vector_ids, source_hashes, applied_hashes = _active_vector_fields(
                    condition, bundle
                )
                trace_id = (
                    f"{run_id}-p{prompt_index:04d}-{condition.name}-"
                    f"{sha256_json([prompt_hash, condition.name, seed, generated_ids])[:12]}"
                )
                rows.append(
                    {
                        "schema_version": "factorial-event-1.0",
                        "trace_id": trace_id,
                        "run_id": run_id,
                        "prompt_id": prompt_record.prompt_id,
                        "prompt_index": prompt_index,
                        "origin_stratum": prompt_record.origin_stratum,
                        "source_id": prompt_record.source_id,
                        "source_file": prompt_record.source_file,
                        "source_record_index": prompt_record.source_record_index,
                        "condition": condition.name,
                        "condition_index": condition_index,
                        "prompt_hash": prompt_hash,
                        "prompt_text": prompt,
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
                        "do_sample": not args.greedy,
                        "temperature": None if args.greedy else args.temperature,
                        "top_p": None if args.greedy else args.top_p,
                        "sampling_seed": seed if not args.greedy else None,
                        "paired_seed": seed,
                        "effective_alphas": condition.effective_alphas,
                        "controller_alphas": condition.controller_alphas,
                        "steering_applied": any(
                            alpha != 0.0 for alpha in condition.controller_alphas.values()
                        ),
                        "vector_mode": condition.vector_mode,
                        "steering_vector_ids": vector_ids,
                        "steering_vector_hashes": source_hashes,
                        "applied_vector_hashes": applied_hashes,
                        "vector_mapping": condition_vector_provenance(condition, bundle),
                    }
                )
    finally:
        controller.clear_prompt_metadata()
        controller.remove()

    _write_jsonl_atomic(args.output, rows)
    manifest = {
        **run_spec,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_source": str(args.prompts),
        "prompt_source_sha256": sha256_file(args.prompts),
        "prompts": len(prompts),
        "conditions_per_prompt": len(conditions),
        "events": len(rows),
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "script_sha256": sha256_file(Path(__file__)),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    write_json_atomic(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

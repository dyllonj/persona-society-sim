"""Compute trait steering vectors via Contrastive Activation Addition (CAA)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts.schema import PromptItem, load_prompt_items
from steering.vector_store import VectorStore, coerce_polarity

DEFAULT_LAYERS: Tuple[int, ...] = ()
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
METHOD_NAME = "caa_ab"
PROMPT_TEMPLATE = (
    "You are rating a persona in a town simulator.\n"
    "Question: {question}\n"
    "Answer: ({letter})"
)


@dataclass(frozen=True)
class ResolvedCAAConfig:
    """Fully resolved inputs for one CAA vector extraction."""

    trait: str
    prompt_file: Path
    output_dir: Path
    model: str
    layers: Tuple[int, ...]
    vector_store_id: str | None
    config_path: Path | None = None
    expected_num_hidden_layers: int | None = None
    polarity: float = 1.0


def _resolve_relative_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_steering_config(config_path: Path | None) -> Mapping[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Steering config must be a mapping: {config_path}")
    return payload


def _coerce_layers(value: Sequence[int] | None, *, source: str) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    try:
        layers = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a list of integer layer ids") from exc
    return tuple(sorted(set(layers)))


def _resolve_trait_prompt_file(
    trait: str,
    trait_cfg: Mapping[str, Any],
    defaults: Mapping[str, Any],
    config_dir: Path,
) -> Path | None:
    prompt_file = trait_cfg.get("prompt_file")
    if prompt_file:
        return _resolve_relative_path(prompt_file, config_dir)

    prompt_dir_value = defaults.get("prompt_dir")
    if not prompt_dir_value:
        return None
    prompt_dir = _resolve_relative_path(prompt_dir_value, config_dir)
    prompt_name = trait_cfg.get("name") or trait
    return prompt_dir / f"{str(prompt_name).lower()}.jsonl"


def resolve_caa_config(
    trait: str,
    *,
    config_path: Path | None = None,
    prompt_file: Path | None = None,
    output_dir: Path | None = None,
    model: str | None = None,
    layers: Sequence[int] | None = None,
    vector_store_id: str | None = None,
) -> ResolvedCAAConfig:
    """Merge CLI inputs with ``configs/steering.layers.yaml`` metadata.

    Explicit CLI values win. When ``--config`` is provided, missing values are
    resolved from ``defaults`` plus the selected ``traits.<trait>`` entry.
    """

    config = _load_steering_config(config_path)
    defaults = config.get("defaults") or {}
    traits_cfg = config.get("traits") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("Steering config defaults must be a mapping")
    if not isinstance(traits_cfg, Mapping):
        raise ValueError("Steering config traits must be a mapping")

    trait_cfg_raw = traits_cfg.get(trait) or {}
    if config_path is not None and not trait_cfg_raw:
        raise ValueError(f"Trait {trait!r} is not defined in {config_path}")
    if not isinstance(trait_cfg_raw, Mapping):
        raise ValueError(f"Steering config for trait {trait!r} must be a mapping")
    trait_cfg: Mapping[str, Any] = trait_cfg_raw

    config_dir = config_path.parent.resolve() if config_path is not None else Path.cwd()

    resolved_prompt_file = prompt_file
    if resolved_prompt_file is None and config_path is not None:
        resolved_prompt_file = _resolve_trait_prompt_file(
            trait, trait_cfg, defaults, config_dir
        )
    if resolved_prompt_file is None:
        raise ValueError(
            "prompt_file is required unless --config provides traits.<trait>.prompt_file "
            "or defaults.prompt_dir"
        )
    if config_path is not None and prompt_file is None:
        resolved_prompt_file = resolved_prompt_file.resolve()

    resolved_output_dir = output_dir
    if resolved_output_dir is None and config_path is not None:
        vector_root = config.get("vector_root")
        if vector_root:
            resolved_output_dir = _resolve_relative_path(vector_root, config_dir)
    if resolved_output_dir is None:
        raise ValueError(
            "output_dir is required unless --config provides vector_root"
        )

    default_model = defaults.get("model")
    resolved_model = model or default_model or DEFAULT_MODEL
    if not isinstance(resolved_model, str) or not resolved_model:
        raise ValueError("Resolved model name must be a non-empty string")

    if layers is not None:
        resolved_layers = _coerce_layers(layers, source="--layers")
    else:
        resolved_layers = _coerce_layers(
            trait_cfg.get("layers") or defaults.get("layers") or DEFAULT_LAYERS,
            source=f"{config_path}: layers" if config_path is not None else "layers",
        )
    if not resolved_layers:
        raise ValueError(
            "You must provide at least one decoder layer via --layers or steering config; "
            "the legacy [12, 16, 20] default was removed."
        )

    yaml_num_layers = defaults.get("num_hidden_layers")
    expected_num_hidden_layers: int | None = None
    if yaml_num_layers is not None and (model is None or model == default_model):
        try:
            expected_num_hidden_layers = int(yaml_num_layers)
        except (TypeError, ValueError) as exc:
            raise ValueError("defaults.num_hidden_layers must be an integer") from exc

    resolved_vector_store_id = vector_store_id or trait_cfg.get("vector_store_id")
    if resolved_vector_store_id is not None:
        resolved_vector_store_id = str(resolved_vector_store_id)
    resolved_polarity = coerce_polarity(trait_cfg.get("polarity", 1.0))

    return ResolvedCAAConfig(
        trait=trait,
        prompt_file=Path(resolved_prompt_file),
        output_dir=Path(resolved_output_dir),
        model=resolved_model,
        layers=resolved_layers,
        vector_store_id=resolved_vector_store_id,
        config_path=config_path.resolve() if config_path is not None else None,
        expected_num_hidden_layers=expected_num_hidden_layers,
        polarity=resolved_polarity,
    )


def validate_model_config(model: AutoModelForCausalLM, resolved: ResolvedCAAConfig) -> None:
    """Validate the loaded model against resolved YAML layer metadata."""

    model_config = getattr(model, "config", None)
    actual_num_layers = getattr(model_config, "num_hidden_layers", None)
    if actual_num_layers is None:
        if resolved.expected_num_hidden_layers is not None:
            raise ValueError(
                "Loaded model config does not expose num_hidden_layers; cannot validate "
                f"against {resolved.config_path}"
            )
        return
    actual_num_layers = int(actual_num_layers)

    if (
        resolved.expected_num_hidden_layers is not None
        and actual_num_layers != resolved.expected_num_hidden_layers
    ):
        raise ValueError(
            "Model/config layer mismatch: "
            f"{resolved.model} reports model.config.num_hidden_layers={actual_num_layers}, "
            f"but {resolved.config_path} declares defaults.num_hidden_layers="
            f"{resolved.expected_num_hidden_layers}."
        )

    invalid_layers = [
        layer for layer in resolved.layers if layer < 0 or layer >= actual_num_layers
    ]
    if invalid_layers:
        raise ValueError(
            f"Layer ids {invalid_layers} are invalid for {resolved.model}; "
            f"valid decoder layer ids are 0..{actual_num_layers - 1}."
        )


def build_prompt(question: str, letter: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip(), letter=letter)


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def high_low_letters(item: PromptItem) -> Tuple[str, str]:
    return ("A", "B") if item.option_a_is_high else ("B", "A")


def _find_answer_token_index(prompt: str, letter: str, offsets: Sequence[Tuple[int, int]]) -> int:
    marker = f"({letter})"
    char_index = prompt.index(marker) + 1  # letter position inside the marker
    for idx, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return idx
    raise ValueError(f"Unable to locate token index for letter {letter} in prompt")


def _validate_answer_token_mask(
    tokenizer,
    token_ids: Sequence[int],
    token_index: int,
    letter: str,
    attention_mask: Sequence[int] | None = None,
) -> None:
    token_str = tokenizer.convert_ids_to_tokens([token_ids[token_index]])[0]
    if letter.lower() not in token_str.lower():
        raise ValueError(
            f"Answer token {token_str!r} at position {token_index} does not align with letter {letter}"
        )
    if attention_mask and attention_mask[token_index] == 0:
        raise ValueError(
            f"Answer token for letter {letter} is masked out by the tokenizer attention mask"
        )


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def directional_agreement(diffs: np.ndarray) -> float:
    """Mean cosine agreement of individual activation diffs with their mean."""

    if diffs.size == 0:
        return 0.0
    matrix = np.asarray(diffs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    mean_vec = matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    if mean_norm == 0.0:
        return 0.0
    cosines = []
    for row in matrix:
        row_norm = float(np.linalg.norm(row))
        if row_norm == 0.0:
            cosines.append(0.0)
        else:
            cosines.append(float(np.dot(row, mean_vec) / (row_norm * mean_norm)))
    return float(np.mean(cosines)) if cosines else 0.0


def separability_d_prime(diffs: np.ndarray) -> float:
    """Projection-space discriminability for high-minus-low activation diffs."""

    if diffs.size == 0:
        return 0.0
    matrix = np.asarray(diffs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    mean_vec = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm == 0.0:
        return 0.0
    direction = mean_vec / norm
    projections = matrix @ direction
    std = float(np.std(projections))
    if std == 0.0:
        return float(np.mean(projections))
    return float(np.mean(projections) / std)


def _existing_layer_vectors(
    vector_root: Path, target_layers: Iterable[int], exclude_id: str
) -> List[Tuple[str, int, np.ndarray]]:
    """Load existing vectors from disk for orthogonality checks."""

    store = VectorStore(vector_root)
    collected: List[Tuple[str, int, np.ndarray]] = []
    for meta_path in vector_root.glob("*.meta.json"):
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            continue
        vector_store_id = metadata.get("vector_store_id") or metadata.get("trait")
        if not vector_store_id or vector_store_id == exclude_id:
            continue
        layers = {entry.get("layer_id") for entry in metadata.get("layers", [])}
        layers &= set(target_layers)
        if not layers:
            continue
        try:
            bundle = store.load(vector_store_id, layers=layers)
        except Exception:
            continue
        trait_name = metadata.get("trait") or vector_store_id
        for layer_id, vec in bundle.vectors.items():
            collected.append((trait_name, layer_id, vec))
    return collected


def enforce_orthogonality(
    candidate_vectors: Dict[int, np.ndarray],
    existing_vectors: Iterable[Tuple[str, int, np.ndarray]],
    *,
    threshold: float = 0.2,
) -> None:
    """Raise if any layer vector is too aligned with existing traits."""

    violations: List[str] = []
    for trait_name, layer_id, existing in existing_vectors:
        if layer_id not in candidate_vectors:
            continue
        similarity = abs(_cosine_similarity(candidate_vectors[layer_id], existing))
        if similarity >= threshold:
            violations.append(
                f"layer {layer_id} overlaps with {trait_name} (|cos|={similarity:.3f} >= {threshold})"
            )
    if violations:
        raise ValueError(
            "Orthogonality check failed: " + "; ".join(sorted(violations))
        )


@torch.no_grad()
def encode_answer_token(
    model: AutoModelForCausalLM,
    tokenizer,
    question: str,
    letter: str,
    layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    prompt = build_prompt(question, letter)
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    if "offset_mapping" not in tokens:
        raise RuntimeError("Tokenizer must support offset mappings; enable the fast tokenizer variant.")
    offsets = tokens.pop("offset_mapping")[0].tolist()
    token_index = _find_answer_token_index(prompt, letter, offsets)
    input_ids = tokens["input_ids"][0].tolist()
    attention_mask = tokens.get("attention_mask")
    _validate_answer_token_mask(
        tokenizer,
        input_ids,
        token_index,
        letter,
        attention_mask[0].tolist() if attention_mask is not None else None,
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    outputs = model(
        **tokens,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    layer_states: Dict[int, torch.Tensor] = {}
    for layer in layers:
        layer_states[layer] = hidden_states[layer][0, token_index, :].detach().cpu()
    return layer_states


def compute_trait_vectors(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[PromptItem],
    layers: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, Dict[str, float]]]:
    accum: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    for item in prompts:
        high_letter, low_letter = high_low_letters(item)
        high_state = encode_answer_token(model, tokenizer, item.question_text, high_letter, layers)
        low_state = encode_answer_token(model, tokenizer, item.question_text, low_letter, layers)
        for layer in layers:
            accum[layer].append(high_state[layer] - low_state[layer])
    vectors: Dict[int, np.ndarray] = {}
    norms: Dict[int, float] = {}
    diagnostics: Dict[int, Dict[str, float]] = {}
    for layer, diffs in accum.items():
        stacked = torch.stack(diffs, dim=0)
        mean_diff = stacked.mean(dim=0)
        norm = float(torch.linalg.norm(mean_diff).item())
        norms[layer] = norm
        diff_array = stacked.float().detach().cpu().numpy()
        diagnostics[layer] = {
            "directional_agreement": directional_agreement(diff_array),
            "separability_d_prime": separability_d_prime(diff_array),
        }
        if norm > 0:
            mean_diff = mean_diff / norm
        vectors[layer] = mean_diff.float().cpu().numpy().astype(np.float32)
    return vectors, norms, diagnostics


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CAA steering vectors from A/B prompt files.")
    parser.add_argument("trait", help="Trait identifier (e.g., E, A, C)")
    parser.add_argument(
        "prompt_file",
        nargs="?",
        type=Path,
        help="Path to trait JSONL prompt file (A/B schema). Optional with --config.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Directory to store vectors and metadata. Optional with --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to steering.layers.yaml for model/layer/prompt/vector metadata.",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        help="Decoder layer ids to sample. Overrides config layers when provided.",
    )
    parser.add_argument("--vector-store-id")
    parser.add_argument(
        "--output-dir",
        dest="output_dir_override",
        type=Path,
        help="Override vector output directory without using the positional output_dir.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved = resolve_caa_config(
        args.trait,
        config_path=args.config,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir_override or args.output_dir,
        model=args.model,
        layers=args.layers,
        vector_store_id=args.vector_store_id,
    )

    model = AutoModelForCausalLM.from_pretrained(
        resolved.model, torch_dtype=torch.float16, device_map="auto"
    )
    validate_model_config(model, resolved)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(resolved.model, use_fast=True)

    prompts = load_prompt_items(resolved.prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {resolved.prompt_file}")

    layers = list(resolved.layers)
    vectors, norms, diagnostics = compute_trait_vectors(model, tokenizer, prompts, layers)
    existing = _existing_layer_vectors(
        resolved.output_dir, layers, resolved.vector_store_id or resolved.trait
    )
    enforce_orthogonality(vectors, existing)

    store = VectorStore(resolved.output_dir)
    metadata = store.save_vectors(
        resolved.trait,
        METHOD_NAME,
        vectors,
        model_name=resolved.model,
        train_set_hash=compute_file_hash(resolved.prompt_file),
        norms=norms,
        hyperparameters={
            "prompt_template": PROMPT_TEMPLATE,
            "token_position": "answer_letter",
            "pooling": "token",
            "difference_equation": "high_minus_low",
            "layers": layers,
        },
        num_train_prompts=len(prompts),
        vector_store_id=resolved.vector_store_id,
        layer_diagnostics=diagnostics,
        polarity=resolved.polarity,
    )
    print(
        f"Saved steering vectors for trait {resolved.trait} with vector_store_id={metadata['vector_store_id']}"
    )
    return 0


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    main()

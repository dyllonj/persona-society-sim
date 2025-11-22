"""Compute trait steering vectors via Contrastive Activation Addition (CAA)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts.schema import PromptItem, load_prompt_items
from steering.vector_store import VectorStore

DEFAULT_LAYERS: Tuple[int, ...] = ()
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
METHOD_NAME = "caa_ab"
PROMPT_TEMPLATE = (
    "You are rating a persona in a town simulator.\n"
    "Question: {question}\n"
    "Answer: ({letter})"
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
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    accum: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    for item in prompts:
        high_letter, low_letter = high_low_letters(item)
        high_state = encode_answer_token(model, tokenizer, item.question_text, high_letter, layers)
        low_state = encode_answer_token(model, tokenizer, item.question_text, low_letter, layers)
        for layer in layers:
            accum[layer].append(high_state[layer] - low_state[layer])
    vectors: Dict[int, np.ndarray] = {}
    norms: Dict[int, float] = {}
    for layer, diffs in accum.items():
        stacked = torch.stack(diffs, dim=0)
        mean_diff = stacked.mean(dim=0)
        norm = float(torch.linalg.norm(mean_diff).item())
        norms[layer] = norm
        if norm > 0:
            mean_diff = mean_diff / norm
        vectors[layer] = mean_diff.cpu().numpy().astype(np.float32)
    return vectors, norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAA steering vectors from A/B prompt files.")
    parser.add_argument("trait", help="Trait identifier (e.g., E, A, C)")
    parser.add_argument("prompt_file", type=Path, help="Path to trait JSONL prompt file (A/B schema)")
    parser.add_argument("output_dir", type=Path, help="Directory to store vectors and metadata")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=list(DEFAULT_LAYERS),
        help="Decoder layer ids to sample (required; no implicit default).",
    )
    parser.add_argument("--vector-store-id")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompts = load_prompt_items(args.prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    layers = sorted(set(args.layers))
    if not layers:
        raise ValueError(
            "You must provide at least one decoder layer via --layers; the legacy [12, 16, 20] default was removed."
        )
    vectors, norms = compute_trait_vectors(model, tokenizer, prompts, layers)
    existing = _existing_layer_vectors(args.output_dir, layers, args.vector_store_id or args.trait)
    enforce_orthogonality(vectors, existing)

    store = VectorStore(args.output_dir)
    metadata = store.save_vectors(
        args.trait,
        METHOD_NAME,
        vectors,
        model_name=args.model,
        train_set_hash=compute_file_hash(args.prompt_file),
        norms=norms,
        hyperparameters={
            "prompt_template": PROMPT_TEMPLATE,
            "token_position": "answer_letter",
            "pooling": "token",
            "difference_equation": "high_minus_low",
            "layers": layers,
        },
        num_train_prompts=len(prompts),
        vector_store_id=args.vector_store_id,
    )
    print(
        f"Saved steering vectors for trait {args.trait} with vector_store_id={metadata['vector_store_id']}"
    )


if __name__ == "__main__":
    main()

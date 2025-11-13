"""Compute trait steering vectors via Contrastive Activation Addition (CAA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LAYERS = [12, 16, 20]
PROMPT_TEMPLATE = (
    "You are rating a persona in a town simulator. Situation: {situation}\n"
    "Candidate response: {response}\nSummarize the activation for steering computations."
)


def load_pairs(path: Path) -> List[Dict[str, str]]:
    pairs = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    if not pairs:
        raise ValueError(f"No prompt pairs found in {path}")
    return pairs


def _format_prompt(item: Dict[str, str], polarity: str) -> str:
    return PROMPT_TEMPLATE.format(situation=item["situation"], response=item[polarity])


@torch.no_grad()
def encode(model, tokenizer, text: str, layers: Iterable[int]) -> Dict[int, torch.Tensor]:
    tokens = tokenizer(text, return_tensors="pt")
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    outputs = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = outputs.hidden_states  # tuple length num_layers+1
    layer_states: Dict[int, torch.Tensor] = {}
    for layer in layers:
        layer_states[layer] = hidden_states[layer].mean(dim=1).squeeze(0).cpu()
    return layer_states


def compute_trait_vectors(
    model, tokenizer, pairs: List[Dict[str, str]], layers: Iterable[int]
) -> Dict[int, np.ndarray]:
    accum: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    for item in pairs:
        pos = encode(model, tokenizer, _format_prompt(item, "positive"), layers)
        neg = encode(model, tokenizer, _format_prompt(item, "negative"), layers)
        for layer in layers:
            accum[layer].append((pos[layer] - neg[layer]).unsqueeze(0))
    diffs = {layer: torch.cat(layer_diffs, dim=0).mean(dim=0) for layer, layer_diffs in accum.items()}
    return {layer: diff.numpy() for layer, diff in diffs.items()}


def save_vectors(vectors: Dict[int, np.ndarray], output_dir: Path, trait: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, vec in vectors.items():
        out_path = output_dir / f"{trait}.layer{layer}.npy"
        np.save(out_path, vec)
        print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAA steering vectors from prompt pairs.")
    parser.add_argument("trait", help="Trait identifier (e.g., E, A, C)")
    parser.add_argument("prompt_file", type=Path, help="Path to trait JSONL prompt pairs")
    parser.add_argument("output_dir", type=Path, help="Directory to store vectors")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", nargs="*", type=int, default=DEFAULT_LAYERS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    pairs = load_pairs(args.prompt_file)
    vectors = compute_trait_vectors(model, tokenizer, pairs, args.layers)
    save_vectors(vectors, args.output_dir, args.trait)


if __name__ == "__main__":
    main()

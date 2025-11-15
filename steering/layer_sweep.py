"""Evaluate steering vector layers on held-out prompts and update metadata."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts.schema import PromptItem, load_prompt_items
from steering.compute_caa import compute_file_hash, encode_answer_token, high_low_letters
from steering.vector_store import VectorStore

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _score_layers(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[PromptItem],
    layers: Sequence[int],
    vector_tensors: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, Dict[str, float]], List[int]]:
    scores: Dict[int, Dict[str, float]] = {
        layer: {"correct": 0.0, "total": 0.0} for layer in layers
    }
    for item in prompts:
        high_letter, low_letter = high_low_letters(item)
        high_state = encode_answer_token(model, tokenizer, item.question_text, high_letter, layers)
        low_state = encode_answer_token(model, tokenizer, item.question_text, low_letter, layers)
        for layer in layers:
            diff = (high_state[layer] - low_state[layer]).to(torch.float32)
            score = torch.dot(vector_tensors[layer], diff)
            if score.item() > 0:
                scores[layer]["correct"] += 1
            scores[layer]["total"] += 1
    metrics: Dict[int, Dict[str, float]] = {}
    best_accuracy = -1.0
    for layer, counts in scores.items():
        total = counts["total"] or 1.0
        accuracy = counts["correct"] / total
        delta = accuracy - 0.5
        metrics[layer] = {
            "accuracy": accuracy,
            "accuracy_delta": delta,
            "total": total,
        }
        best_accuracy = max(best_accuracy, accuracy)
    tol = 1e-6
    best_layers = [
        layer for layer, data in metrics.items() if abs(data["accuracy"] - best_accuracy) <= tol
    ]
    return metrics, best_layers


def run_sweep(
    trait: str,
    vector_root: Path,
    vector_store_id: str,
    prompt_file: Path,
    model_name: str,
) -> Dict[str, object]:
    store = VectorStore(vector_root)
    bundle = store.load(vector_store_id)
    if bundle.trait != trait:
        raise ValueError(
            f"Vector store trait mismatch: expected {trait}, found {bundle.trait}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompts = load_prompt_items(prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")

    layers = sorted(bundle.vectors.keys())
    vector_tensors = {
        layer: torch.from_numpy(vec).to(torch.float32)
        for layer, vec in bundle.vectors.items()
    }
    metrics, best_layers = _score_layers(model, tokenizer, prompts, layers, vector_tensors)

    metadata = bundle.metadata
    metadata.setdefault("dataset", {})
    metadata["dataset"]["eval_size"] = len(prompts)
    metadata["dataset"]["eval_hash"] = compute_file_hash(prompt_file)
    metadata["preferred_layers"] = best_layers
    metadata["layer_sweep"] = {
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "scores": [
            {
                "layer_id": layer,
                "accuracy": metrics[layer]["accuracy"],
                "accuracy_delta": metrics[layer]["accuracy_delta"],
                "total": metrics[layer]["total"],
            }
            for layer in layers
        ],
    }
    for layer_entry in metadata.get("layers", []):
        layer_id = layer_entry["layer_id"]
        if layer_id in metrics:
            layer_entry["accuracy"] = metrics[layer_id]["accuracy"]
            layer_entry["accuracy_delta"] = metrics[layer_id]["accuracy_delta"]
    store.write_metadata(trait, metadata)
    return {
        "best_layers": best_layers,
        "metrics": metrics,
        "metadata_path": store.metadata_path(trait),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a held-out layer sweep for steering vectors.")
    parser.add_argument("trait", help="Trait identifier (e.g., E, A, C)")
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("vector_root", type=Path, help="Directory containing vector bundles")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vector-store-id")
    args = parser.parse_args()

    sweep_id = args.vector_store_id or args.trait
    result = run_sweep(
        args.trait,
        args.vector_root,
        sweep_id,
        args.prompt_file,
        args.model,
    )
    print(
        f"Layer sweep complete for trait {args.trait}. Best layers: {result['best_layers']}"
    )
    print(f"Metadata updated at {result['metadata_path']}")


if __name__ == "__main__":
    main()

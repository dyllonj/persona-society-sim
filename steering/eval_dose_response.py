"""Quick evaluation harness for steering dose-response curves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.hooks import SteeringController
from steering.vector_store import VectorStore

PROMPT = "Share how you would behave at a lively town gathering."  # TODO: replace with behavior probes


def run_curve(
    model_name: str, trait: str, vector_store_id: str, alphas: List[float], layers: List[int]
) -> Dict[float, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    store = VectorStore(Path("data/vectors"))
    vectors = store.load(vector_store_id, layers)
    controller = SteeringController(model, {trait: {layer: torch.tensor(vec) for layer, vec in vectors.items()}})
    controller.register()

    outputs: Dict[float, str] = {}
    for alpha in alphas:
        controller.set_alphas({trait: alpha})
        tokens = tokenizer(PROMPT, return_tensors="pt").to(model.device)
        generated = model.generate(**tokens, max_new_tokens=80)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs[alpha] = text
    controller.remove()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep steering coefficients and log generations.")
    parser.add_argument("vector_store_id")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3-8b-instruct")
    parser.add_argument("--layers", nargs="*", type=int, default=[12, 16, 20])
    parser.add_argument("--alphas", nargs="*", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    args = parser.parse_args()

    outputs = run_curve(args.model, args.trait, args.vector_store_id, args.alphas, args.layers)
    for alpha, text in outputs.items():
        print(f"\n[alpha={alpha}]\n{text}\n")


if __name__ == "__main__":
    main()

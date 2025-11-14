#!/usr/bin/env python3
"""Benchmark sequential vs batched steering additions."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
from torch import nn

from steering.hooks import SteeringController


class BenchmarkLayer(nn.Module):
    def forward(self, hidden_states):  # noqa: D401
        return hidden_states


class BenchmarkModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(BenchmarkLayer() for _ in range(num_layers))

    def forward(self, inputs_embeds):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        return x


def _torch_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(batch_size: int, hidden_size: int, num_layers: int, repeats: int) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BenchmarkModel(num_layers).to(device)
    torch.manual_seed(0)
    trait_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
    for trait_idx in range(4):
        per_layer = {}
        for layer_idx in range(num_layers):
            per_layer[layer_idx] = torch.randn(hidden_size)
        trait_vectors[f"trait_{trait_idx}"] = per_layer

    controller = SteeringController(model, trait_vectors)
    controller.register()

    inputs = torch.zeros(batch_size, 1, hidden_size, device=device)
    batched_alphas: List[Dict[str, float]] = []
    for idx in range(batch_size):
        sample_alphas = {}
        for trait_idx in range(4):
            sample_alphas[f"trait_{trait_idx}"] = torch.sin(torch.tensor(idx + trait_idx)).item()
        batched_alphas.append(sample_alphas)

    sequential_start = time.perf_counter()
    for _ in range(repeats):
        for sample_idx, alphas in enumerate(batched_alphas):
            controller.set_alphas(alphas)
            model(inputs_embeds=inputs[sample_idx : sample_idx + 1])
    _torch_synchronize()
    sequential_time = time.perf_counter() - sequential_start

    controller.set_batched_alphas(batched_alphas)
    batched_start = time.perf_counter()
    for _ in range(repeats):
        model(inputs_embeds=inputs)
    _torch_synchronize()
    batched_time = time.perf_counter() - batched_start
    controller.clear_batched_alphas()

    controller.remove()

    return {
        "sequential": sequential_time,
        "batched": batched_time,
        "speedup": sequential_time / batched_time if batched_time > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched steering")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    timings = benchmark(args.batch_size, args.hidden_size, args.layers, args.repeats)
    print("Sequential time: {:.4f}s".format(timings["sequential"]))
    print("Batched time:    {:.4f}s".format(timings["batched"]))
    print("Speedup:         {:.2f}x".format(timings["speedup"]))


if __name__ == "__main__":
    main()

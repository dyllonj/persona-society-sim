"""Runtime steering hooks for applying persona vectors during generation."""

from __future__ import annotations

from typing import Dict

import torch

LayerVectors = Dict[str, Dict[int, torch.Tensor]]


class SteeringController:
    """Registers forward hooks that add alpha * vector to residual streams."""

    def __init__(self, model, trait_vectors: LayerVectors):
        self.model = model
        self.trait_vectors = {
            trait: {layer: torch.tensor(vec) if not isinstance(vec, torch.Tensor) else vec for layer, vec in by_layer.items()}
            for trait, by_layer in trait_vectors.items()
        }
        self.alphas = {trait: 0.0 for trait in self.trait_vectors}
        self._handles = []
        self.enabled = True

    def set_alphas(self, alphas: Dict[str, float]) -> None:
        self.alphas.update(alphas)

    def register(self) -> None:
        layers_module = getattr(self.model, "model", None)
        if layers_module is None or not hasattr(layers_module, "layers"):
            raise ValueError("Model does not expose decoder layers via model.layers")
        for idx, layer_module in enumerate(layers_module.layers):
            if idx in self.needed_layers:
                handle = layer_module.register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def needed_layers(self):
        needed = set()
        for by_layer in self.trait_vectors.values():
            needed.update(by_layer.keys())
        return needed

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):  # pylint: disable=unused-argument
            if not self.enabled:
                return output
            base = output[0] if isinstance(output, tuple) else output
            delta = None
            for trait, by_layer in self.trait_vectors.items():
                coeff = self.alphas.get(trait, 0.0)
                if coeff == 0.0:
                    continue
                vec = by_layer.get(layer_idx)
                if vec is None:
                    continue
                if vec.device != base.device:
                    by_layer[layer_idx] = vec = vec.to(base.device)
                addition = coeff * vec
                delta = addition if delta is None else delta + addition
            if delta is None:
                return output
            if base.dim() == 3:
                base = base + delta.view(1, 1, -1)
            else:
                base = base + delta
            if isinstance(output, tuple):
                return (base,) + output[1:]
            return base

        return hook

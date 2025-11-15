"""Runtime steering hooks for applying persona vectors during generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency for inference
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

TorchTensor = torch.Tensor if torch is not None else Any  # type: ignore
LayerVectors = Dict[str, Dict[int, TorchTensor]]


class SteeringController:
    """Registers forward hooks that add alpha * vector to residual streams."""

    def __init__(self, model, trait_vectors: LayerVectors):
        if torch is None:
            raise ModuleNotFoundError("torch is required for SteeringController")
        self.model = model
        self.trait_vectors = {
            trait: {layer: torch.tensor(vec) if not isinstance(vec, torch.Tensor) else vec for layer, vec in by_layer.items()}
            for trait, by_layer in trait_vectors.items()
        }
        self.alphas = {trait: 0.0 for trait in self.trait_vectors}
        self._handles = []
        self.enabled = True
        self._batched_alphas: Optional[List[Dict[str, float]]] = None
        self._batched_cache: Dict[int, torch.Tensor] = {}

    def set_alphas(self, alphas: Dict[str, float]) -> None:
        self._batched_alphas = None
        self._batched_cache.clear()
        self.alphas.update(alphas)

    def set_batched_alphas(self, batched_alphas: List[Dict[str, float]]) -> None:
        """Use a different steering vector per batch element."""

        self._batched_alphas = batched_alphas
        self._batched_cache.clear()

    def clear_batched_alphas(self) -> None:
        self._batched_alphas = None
        self._batched_cache.clear()

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
            if self._batched_alphas is not None:
                delta = self._batched_delta(layer_idx, base)
                if delta is None:
                    return output
                if base.dim() == 3:
                    base = base + delta[:, None, :]
                else:
                    base = base + delta
            else:
                delta = self._unbatched_delta(layer_idx, base)
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

    def _unbatched_delta(self, layer_idx: int, base: torch.Tensor) -> Optional[torch.Tensor]:
        delta = None
        for trait, by_layer in self.trait_vectors.items():
            coeff = self.alphas.get(trait, 0.0)
            if coeff == 0.0:
                continue
            vec = self._vector_for_layer(by_layer, layer_idx, base)
            if vec is None:
                continue
            addition = coeff * vec
            delta = addition if delta is None else delta + addition
        return delta

    def _batched_delta(self, layer_idx: int, base: torch.Tensor) -> Optional[torch.Tensor]:
        cached = self._batched_cache.get(layer_idx)
        if cached is not None and cached.device == base.device and cached.dtype == base.dtype:
            return cached

        per_sample = []
        template = None
        for alphas in self._batched_alphas or []:
            sample_delta = None
            for trait, coeff in alphas.items():
                if coeff == 0.0:
                    continue
                by_layer = self.trait_vectors.get(trait)
                if by_layer is None:
                    continue
                vec = self._vector_for_layer(by_layer, layer_idx, base)
                if vec is None:
                    continue
                addition = coeff * vec
                sample_delta = addition if sample_delta is None else sample_delta + addition
            if sample_delta is not None and template is None:
                template = sample_delta
            per_sample.append(sample_delta)

        if not per_sample:
            return None

        if template is None:
            return None

        stacked = []
        for sample_delta in per_sample:
            if sample_delta is None:
                stacked.append(torch.zeros_like(template))
            else:
                stacked.append(sample_delta)

        delta = torch.stack(stacked, dim=0)
        self._batched_cache[layer_idx] = delta
        return delta

    def _vector_for_layer(self, by_layer: Dict[int, torch.Tensor], layer_idx: int, base: torch.Tensor) -> Optional[torch.Tensor]:
        vec = by_layer.get(layer_idx)
        if vec is None:
            return None
        if vec.device != base.device or vec.dtype != base.dtype:
            by_layer[layer_idx] = vec = vec.to(device=base.device, dtype=base.dtype)
        return vec

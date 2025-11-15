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

    def __init__(
        self,
        model,
        trait_vectors: LayerVectors,
        *,
        vector_norms: Optional[Dict[str, Dict[int, float]]] = None,
    ):
        if torch is None:
            raise ModuleNotFoundError("torch is required for SteeringController")
        self.model = model
        provided_norms = vector_norms or {}
        self.vector_norms: Dict[str, Dict[int, float]] = {
            trait: dict(per_layer) for trait, per_layer in provided_norms.items()
        }
        self.trait_vectors: LayerVectors = {}
        for trait, by_layer in trait_vectors.items():
            normalized_layers: Dict[int, TorchTensor] = {}
            for layer, vec in by_layer.items():
                tensor = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                if hasattr(torch.linalg, "norm"):
                    norm_tensor = torch.linalg.norm(tensor)
                else:  # pragma: no cover - legacy torch fallback
                    norm_tensor = tensor.norm()
                norm_value = float(norm_tensor.item()) if hasattr(norm_tensor, "item") else float(norm_tensor)
                if norm_value > 0:
                    tensor = tensor / norm_value
                normalized_layers[layer] = tensor
                self.vector_norms.setdefault(trait, {})[layer] = norm_value
            if normalized_layers:
                self.trait_vectors[trait] = normalized_layers
        self.alphas = {trait: 0.0 for trait in self.trait_vectors}
        self._handles = []
        self.enabled = True
        self._batched_alphas: Optional[List[Dict[str, float]]] = None
        self._batched_cache: Dict[int, torch.Tensor] = {}
        self._prompt_length: Optional[int] = None
        self._batched_prompt_lengths: Optional[List[int]] = None
        self._prompt_hook_calls_remaining = 0
        self._batched_prompt_hook_calls_remaining = 0

    def set_alphas(self, alphas: Dict[str, float], prompt_length: Optional[int] = None) -> None:
        self._batched_alphas = None
        self._batched_cache.clear()
        self._prompt_length = prompt_length
        self._batched_prompt_lengths = None
        self._batched_prompt_hook_calls_remaining = 0
        self._prompt_hook_calls_remaining = len(self.needed_layers) if prompt_length else 0
        self.alphas.update(alphas)

    def set_batched_alphas(
        self, batched_alphas: List[Dict[str, float]], prompt_lengths: Optional[List[int]] = None
    ) -> None:
        """Use a different steering vector per batch element."""

        if prompt_lengths is not None and len(prompt_lengths) != len(batched_alphas):
            raise ValueError("prompt_lengths must match the number of batched alpha sets")
        self._batched_alphas = batched_alphas
        self._batched_cache.clear()
        self._prompt_length = None
        self._batched_prompt_lengths = list(prompt_lengths) if prompt_lengths is not None else None
        has_prompt_lengths = bool(self._batched_prompt_lengths) and any(
            length > 0 for length in (self._batched_prompt_lengths or [])
        )
        self._batched_prompt_hook_calls_remaining = len(self.needed_layers) if has_prompt_lengths else 0
        self._prompt_hook_calls_remaining = 0

    def clear_batched_alphas(self) -> None:
        self._batched_alphas = None
        self._batched_cache.clear()
        self._batched_prompt_lengths = None
        self._batched_prompt_hook_calls_remaining = 0

    def clear_prompt_metadata(self) -> None:
        self._prompt_length = None
        self._batched_prompt_lengths = None
        self._prompt_hook_calls_remaining = 0
        self._batched_prompt_hook_calls_remaining = 0

    def register(self) -> None:
        layers_module = getattr(self.model, "model", None)
        if layers_module is None or not hasattr(layers_module, "layers"):
            raise ValueError("Model does not expose decoder layers via model.layers")
        needed = set(self.needed_layers)
        for idx, layer_module in enumerate(layers_module.layers):
            if idx in needed:
                handle = layer_module.register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def needed_layers(self) -> List[int]:
        needed = set()
        for by_layer in self.trait_vectors.values():
            needed.update(by_layer.keys())
        return sorted(needed)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):  # pylint: disable=unused-argument
            if not self.enabled:
                return output
            base = output[0] if isinstance(output, tuple) else output
            if self._batched_alphas is not None:
                delta = self._batched_delta(layer_idx, base)
                if delta is None:
                    return output
                base = self._apply_batched_delta(base, delta)
            else:
                delta = self._unbatched_delta(layer_idx, base)
                if delta is None:
                    return output
                base = self._apply_unbatched_delta(base, delta)
            if isinstance(output, tuple):
                return (base,) + output[1:]
            return base

        return hook

    def _apply_unbatched_delta(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if base.dim() == 3:
            addition = delta.view(1, 1, -1)
            mask = self._build_unbatched_mask(base)
            if mask is not None:
                addition = addition * mask
            return base + addition
        if base.dim() == 2:
            addition = delta.view(1, -1)
            mask = self._build_unbatched_mask(base)
            if mask is not None:
                addition = addition * mask
            return base + addition
        return base + delta

    def _apply_batched_delta(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if base.dim() == 3:
            addition = delta[:, None, :]
            mask = self._build_batched_mask(base)
            if mask is not None:
                addition = addition * mask
            return base + addition
        return base + delta

    def _build_unbatched_mask(self, base: torch.Tensor) -> Optional[torch.Tensor]:
        if self._prompt_length is None or self._prompt_hook_calls_remaining == 0:
            return None
        seq_dim = 1 if base.dim() == 3 else 0
        seq_len = base.shape[seq_dim]
        clip = max(0, min(self._prompt_length, seq_len))
        if clip == 0:
            return None
        if base.dim() == 3:
            mask = torch.ones((1, seq_len, 1), device=base.device, dtype=base.dtype)
            mask[:, :clip, :] = 0
        elif base.dim() == 2:
            mask = torch.ones((seq_len, 1), device=base.device, dtype=base.dtype)
            mask[:clip, :] = 0
        else:
            return None
        self._prompt_hook_calls_remaining = max(0, self._prompt_hook_calls_remaining - 1)
        return mask

    def _build_batched_mask(self, base: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            self._batched_prompt_lengths is None
            or base.dim() != 3
            or self._batched_prompt_hook_calls_remaining == 0
        ):
            return None
        batch_size, seq_len, _ = base.shape
        mask = torch.ones((batch_size, seq_len, 1), device=base.device, dtype=base.dtype)
        has_prompt_tokens = False
        for idx in range(batch_size):
            length = 0
            if idx < len(self._batched_prompt_lengths):
                length = self._batched_prompt_lengths[idx]
            clip = max(0, min(length, seq_len))
            if clip > 0:
                mask[idx, :clip, :] = 0
                has_prompt_tokens = True
        if not has_prompt_tokens:
            return None
        self._batched_prompt_hook_calls_remaining = max(
            0, self._batched_prompt_hook_calls_remaining - 1
        )
        return mask

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

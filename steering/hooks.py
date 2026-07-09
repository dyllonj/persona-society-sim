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
            layer_vectors: Dict[int, TorchTensor] = {}
            for layer, vec in by_layer.items():
                tensor = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                if hasattr(torch.linalg, "norm"):
                    norm_tensor = torch.linalg.norm(tensor)
                else:  # pragma: no cover - legacy torch fallback
                    norm_tensor = tensor.norm()
                norm_value = float(norm_tensor.item()) if hasattr(norm_tensor, "item") else float(norm_tensor)
                layer_vectors[layer] = tensor
                self.vector_norms.setdefault(trait, {})[layer] = norm_value
            if layer_vectors:
                self.trait_vectors[trait] = layer_vectors
        self.alphas = {trait: 0.0 for trait in self.trait_vectors}
        self._handles = []
        self.enabled = True
        self._batched_alphas: Optional[List[Dict[str, float]]] = None
        self._batched_cache: Dict[int, torch.Tensor] = {}
        self._prompt_mask: Optional[torch.Tensor] = None
        self._batched_prompt_masks: Optional[torch.Tensor] = None
        self._prompt_hook_calls_remaining = 0
        self._batched_prompt_hook_calls_remaining = 0

    def set_alphas(
        self,
        alphas: Dict[str, float],
        prompt_length: Optional[int] = None,
        *,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if prompt_length is not None and prompt_mask is not None:
            raise ValueError("pass prompt_length or prompt_mask, not both")
        self._batched_alphas = None
        self._batched_cache.clear()
        if prompt_mask is not None:
            normalized_mask = torch.as_tensor(prompt_mask, dtype=torch.bool).flatten()
        elif prompt_length:
            normalized_mask = torch.ones(int(prompt_length), dtype=torch.bool)
        else:
            normalized_mask = None
        self._prompt_mask = normalized_mask
        self._batched_prompt_masks = None
        self._batched_prompt_hook_calls_remaining = 0
        self._prompt_hook_calls_remaining = (
            len(self.needed_layers) if normalized_mask is not None else 0
        )
        self.alphas.update(alphas)

    def set_batched_alphas(
        self,
        batched_alphas: List[Dict[str, float]],
        prompt_lengths: Optional[List[int]] = None,
        *,
        prompt_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """Use a different steering vector per batch element."""

        if prompt_lengths is not None and prompt_masks is not None:
            raise ValueError("pass prompt_lengths or prompt_masks, not both")
        if prompt_lengths is not None and len(prompt_lengths) != len(batched_alphas):
            raise ValueError("prompt_lengths must match the number of batched alpha sets")
        normalized_masks: Optional[torch.Tensor]
        if prompt_masks is not None:
            normalized_masks = torch.as_tensor(prompt_masks, dtype=torch.bool)
            if normalized_masks.ndim != 2 or normalized_masks.shape[0] != len(
                batched_alphas
            ):
                raise ValueError(
                    "prompt_masks must have shape [len(batched_alphas), sequence]"
                )
        elif prompt_lengths is not None:
            max_length = max(prompt_lengths, default=0)
            normalized_masks = torch.zeros(
                (len(prompt_lengths), max_length), dtype=torch.bool
            )
            for idx, length in enumerate(prompt_lengths):
                normalized_masks[idx, : max(0, int(length))] = True
        else:
            normalized_masks = None
        self._batched_alphas = batched_alphas
        self._batched_cache.clear()
        self._prompt_mask = None
        self._batched_prompt_masks = normalized_masks
        self._batched_prompt_hook_calls_remaining = (
            len(self.needed_layers) if normalized_masks is not None else 0
        )
        self._prompt_hook_calls_remaining = 0

    def clear_batched_alphas(self) -> None:
        self._batched_alphas = None
        self._batched_cache.clear()
        self._batched_prompt_masks = None
        self._batched_prompt_hook_calls_remaining = 0

    def clear_prompt_metadata(self) -> None:
        self._prompt_mask = None
        self._batched_prompt_masks = None
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
        if self._prompt_mask is None or self._prompt_hook_calls_remaining == 0:
            return None
        seq_dim = 1 if base.dim() == 3 else 0
        seq_len = base.shape[seq_dim]
        prompt_mask = self._prompt_mask.to(base.device)
        if prompt_mask.numel() < seq_len:
            prompt_mask = torch.nn.functional.pad(
                prompt_mask, (0, seq_len - prompt_mask.numel()), value=False
            )
        prompt_mask = prompt_mask[:seq_len]
        addition_mask = (~prompt_mask).to(dtype=base.dtype)
        if base.dim() == 3:
            mask = addition_mask.view(1, seq_len, 1)
        elif base.dim() == 2:
            mask = addition_mask.view(seq_len, 1)
        else:
            return None
        self._prompt_hook_calls_remaining = max(0, self._prompt_hook_calls_remaining - 1)
        return mask

    def _build_batched_mask(self, base: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            self._batched_prompt_masks is None
            or base.dim() != 3
            or self._batched_prompt_hook_calls_remaining == 0
        ):
            return None
        batch_size, seq_len, _ = base.shape
        prompt_masks = self._batched_prompt_masks.to(base.device)
        if prompt_masks.shape[1] < seq_len:
            prompt_masks = torch.nn.functional.pad(
                prompt_masks, (0, seq_len - prompt_masks.shape[1]), value=False
            )
        prompt_masks = prompt_masks[:batch_size, :seq_len]
        mask = (~prompt_masks).to(dtype=base.dtype).unsqueeze(-1)
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

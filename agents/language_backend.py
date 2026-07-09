"""Language backend abstractions for agents."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Dict, List, Mapping, Optional, Union, cast

try:  # pragma: no cover - optional dependency for inference
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore
try:  # pragma: no cover - optional dependency for inference
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    hf_logging = None  # type: ignore

from steering.hooks import SteeringController
from steering.per_trait_strength import resolve_per_trait_strength


@dataclass
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int
    input_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    generated_ids: List[int] = field(default_factory=list)
    effective_alphas: Dict[str, float] = field(default_factory=dict)
    steering_applied: bool = False
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    do_sample: Optional[bool] = None
    sampling_seed: Optional[int] = None
    steering_vector_ids: Dict[str, str] = field(default_factory=dict)
    steering_vector_hashes: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class BatchGenerationRequest:
    """A single generation request in a batch."""
    prompt: str
    max_new_tokens: int
    alphas: Dict[str, float]
    sampling_seed: Optional[int] = None


class LanguageBackend:
    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        alpha_strength: float = 1.0,
        per_trait_strength: Optional[Mapping[str, object]] = None,
        suppress_alphas: bool = False,
        do_sample: bool = False,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.alpha_strength = float(alpha_strength)
        self.per_trait_strength = dict(per_trait_strength or {})
        self.suppress_alphas = suppress_alphas
        self.do_sample = bool(do_sample)

    def _scale_alphas(self, alphas: Dict[str, float]) -> Dict[str, float]:
        if self.suppress_alphas:
            return {trait: 0.0 for trait in alphas.keys()}
        return resolve_per_trait_strength(
            alphas,
            per_trait_strength=self.per_trait_strength,
            global_strength=self.alpha_strength,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        alphas: Dict[str, float],
        *,
        sampling_seed: Optional[int] = None,
    ) -> GenerationResult:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:  # pragma: no cover - interface
        """Generate responses for multiple prompts in parallel."""
        raise NotImplementedError

    def layers_used(self) -> List[int]:  # pragma: no cover - interface
        return []


MemoryMap = Dict[Union[int, str], str]


class HFBackend(LanguageBackend):
    def __init__(
        self,
        model_name: str,
        trait_vectors: Dict[str, Dict[int, torch.Tensor]],
        vector_norms: Optional[Dict[str, Dict[int, float]]] = None,
        vector_artifacts: Optional[Dict[str, Dict[str, object]]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_quantization: bool = False,
        alpha_strength: float = 1.0,
        per_trait_strength: Optional[Mapping[str, object]] = None,
        max_gpu_memory_gb: Optional[float] = None,
        max_cpu_memory_gb: Optional[float] = None,
        offload_folder: Optional[str] = None,
        suppress_alphas: bool = False,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        do_sample: bool = False,
    ):
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError("torch and transformers are required for HFBackend")
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            alpha_strength=alpha_strength,
            per_trait_strength=per_trait_strength,
            suppress_alphas=suppress_alphas,
            do_sample=do_sample,
        )
        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision or model_revision
        self.use_quantization = bool(use_quantization)
        self._generation_lock = RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=self.tokenizer_revision,
        )
        # Reduce Transformers log verbosity (e.g., pad_token warnings)
        if hf_logging is not None:
            try:
                hf_logging.set_verbosity_error()
            except Exception:
                pass

        max_memory = self._build_max_memory_limits(max_gpu_memory_gb, max_cpu_memory_gb)
        offload_dir: Optional[Path] = None
        if offload_folder:
            offload_dir = Path(offload_folder)
            offload_dir.mkdir(parents=True, exist_ok=True)

        model_kwargs: Dict[str, object] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if model_revision:
            model_kwargs["revision"] = model_revision
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if offload_dir:
            model_kwargs["offload_folder"] = str(offload_dir)

        # Load model with optional quantization
        if use_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if not suppress_alphas and not trait_vectors:
            raise ValueError(
                "Steering is enabled but no trait vectors were loaded; refusing to run "
                "with telemetry-only alphas"
            )
        self.vector_artifacts = {
            trait: dict(metadata)
            for trait, metadata in (vector_artifacts or {}).items()
        }
        self._validate_trait_vectors(trait_vectors)

        provided_norms = vector_norms or {}
        self.vector_norms: Dict[str, Dict[int, float]] = {
            trait: dict(per_layer) for trait, per_layer in provided_norms.items()
        }
        self.controller: Optional[SteeringController]
        if suppress_alphas or not trait_vectors:
            self.controller = None
        else:
            self.controller = SteeringController(
                self.model,
                trait_vectors,
                vector_norms=self.vector_norms,
            )
            self.controller.register()

    def _validate_trait_vectors(
        self, trait_vectors: Dict[str, Dict[int, torch.Tensor]]
    ) -> None:
        """Fail before inference when vectors do not match the loaded model."""

        config = self.model.config
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config() if callable(get_text_config) else config
        hidden_size = int(getattr(text_config, "hidden_size"))
        n_layers = int(getattr(text_config, "num_hidden_layers"))
        for trait, by_layer in trait_vectors.items():
            if not by_layer:
                raise ValueError(f"Trait {trait} has no loaded steering layers")
            artifact = self.vector_artifacts.get(trait, {})
            artifact_model = artifact.get("model_name")
            if artifact_model and str(artifact_model) != self.model_name:
                raise ValueError(
                    f"Trait {trait} vector model mismatch: artifact={artifact_model!r}, "
                    f"runtime={self.model_name!r}"
                )
            for layer, vector in by_layer.items():
                if not 0 <= int(layer) < n_layers:
                    raise ValueError(
                        f"Trait {trait} layer {layer} is outside model range [0, {n_layers - 1}]"
                    )
                if vector.ndim != 1 or int(vector.shape[0]) != hidden_size:
                    raise ValueError(
                        f"Trait {trait} layer {layer} vector shape {tuple(vector.shape)} "
                        f"does not match hidden size {hidden_size}"
                    )

    @property
    def steering_vector_ids(self) -> Dict[str, str]:
        return {
            trait: str(metadata.get("vector_store_id") or trait)
            for trait, metadata in self.vector_artifacts.items()
        }

    @property
    def steering_vector_hashes(self) -> Dict[str, Dict[str, str]]:
        hashes: Dict[str, Dict[str, str]] = {}
        for trait, metadata in self.vector_artifacts.items():
            per_layer = metadata.get("vector_hashes") or {}
            if isinstance(per_layer, dict):
                hashes[trait] = {
                    str(layer): str(digest) for layer, digest in per_layer.items()
                }
        return hashes

    @staticmethod
    def _clean_memory_value(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0:
            return None
        return numeric

    @classmethod
    def _build_max_memory_limits(
        cls, max_gpu_memory_gb: Optional[float], max_cpu_memory_gb: Optional[float]
    ) -> Optional[MemoryMap]:
        gpu_limit = cls._clean_memory_value(max_gpu_memory_gb)
        cpu_limit = cls._clean_memory_value(max_cpu_memory_gb)
        limits: MemoryMap = {}
        if gpu_limit is not None and torch.cuda is not None and torch.cuda.is_available():
            gpu_count = max(1, torch.cuda.device_count())
            formatted = f"{gpu_limit:.2f}GiB"
            for idx in range(gpu_count):
                limits[idx] = formatted
        if cpu_limit is not None:
            limits["cpu"] = f"{cpu_limit:.2f}GiB"
        return limits or None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        alphas: Dict[str, float],
        *,
        sampling_seed: Optional[int] = None,
    ) -> GenerationResult:
        with self._generation_lock:
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            scaled_alphas = self._scale_alphas(alphas)
            effective_alphas = self._effective_alphas(scaled_alphas)
            if self.controller:
                self.controller.set_alphas(
                    effective_alphas,
                    prompt_mask=torch.ones_like(
                        tokens["input_ids"][0], dtype=torch.bool
                    ),
                )
            with self._seeded_rng(sampling_seed), torch.inference_mode():
                try:
                    output = self.model.generate(
                        **tokens,
                        **self._generation_kwargs(max_new_tokens),
                    )
                finally:
                    if self.controller:
                        self.controller.clear_prompt_metadata()
        tokens_in = tokens["input_ids"].shape[-1]
        generated = output[0][tokens_in:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        tokens_out = generated.shape[-1]
        return GenerationResult(
            text=decoded,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            input_ids=tokens["input_ids"][0].detach().cpu().tolist(),
            attention_mask=tokens["attention_mask"][0].detach().cpu().tolist(),
            generated_ids=generated.detach().cpu().tolist(),
            effective_alphas=effective_alphas,
            steering_applied=bool(
                self.controller
                and any(abs(value) > 0.0 for value in effective_alphas.values())
            ),
            model_id=self.model_name,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            dtype=self._model_dtype_name(),
            quantization="nf4" if self.use_quantization else None,
            do_sample=self.do_sample,
            sampling_seed=sampling_seed if self.do_sample else None,
            steering_vector_ids=self.steering_vector_ids,
            steering_vector_hashes=self.steering_vector_hashes,
        )

    def _generation_kwargs(self, max_new_tokens: int) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs

    @contextmanager
    def _seeded_rng(self, sampling_seed: Optional[int]):
        if sampling_seed is None or not self.do_sample:
            yield
            return
        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda is not None and torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(sampling_seed))
            if devices:
                torch.cuda.manual_seed_all(int(sampling_seed))
            yield

    def _effective_alphas(self, scaled_alphas: Dict[str, float]) -> Dict[str, float]:
        if not self.controller:
            return {trait: 0.0 for trait in scaled_alphas}
        active_traits = set(self.controller.trait_vectors)
        return {
            trait: float(value) if trait in active_traits else 0.0
            for trait, value in scaled_alphas.items()
        }

    def _model_dtype_name(self) -> str:
        try:
            return str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except (StopIteration, AttributeError):
            return "unknown"

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:
        """Generate responses for multiple prompts in parallel using batched inference."""
        if not requests:
            return []
        if self.do_sample:
            # Per-request seeds are part of the research record. HF generation
            # does not expose independent RNG streams for each batch row, so
            # preserve correctness until a custom sampling loop is introduced.
            return [
                self.generate(
                    request.prompt,
                    request.max_new_tokens,
                    request.alphas,
                    sampling_seed=request.sampling_seed,
                )
                for request in requests
            ]

        grouped: Dict[int, List[tuple[int, BatchGenerationRequest]]] = defaultdict(list)
        for idx, req in enumerate(requests):
            grouped[req.max_new_tokens].append((idx, req))

        results: List[Optional[GenerationResult]] = [None] * len(requests)

        for max_tokens, bucket in grouped.items():
            prompts = [req.prompt for _, req in bucket]
            with self._generation_lock:
                previous_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                try:
                    tokens = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.model.device)
                finally:
                    self.tokenizer.padding_side = previous_padding_side

                input_lengths = (tokens["attention_mask"].sum(dim=1)).tolist()
                padded_width = int(tokens["input_ids"].shape[1])
                batched_alphas = [
                    self._effective_alphas(self._scale_alphas(req.alphas))
                    for _, req in bucket
                ]
                if self.controller:
                    self.controller.set_batched_alphas(
                        batched_alphas,
                        prompt_masks=torch.ones_like(
                            tokens["input_ids"], dtype=torch.bool
                        ),
                    )

                with torch.inference_mode():
                    try:
                        outputs = self.model.generate(
                            **tokens,
                            **self._generation_kwargs(max_tokens),
                        )
                    finally:
                        if self.controller:
                            self.controller.clear_prompt_metadata()
                            self.controller.clear_batched_alphas()

            for row_idx, ((original_idx, req), output, input_len) in enumerate(
                zip(bucket, outputs, input_lengths, strict=True)
            ):
                generated = output[padded_width : padded_width + req.max_new_tokens]
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                tokens_out = generated.shape[-1]
                effective_alphas = batched_alphas[row_idx]
                results[original_idx] = GenerationResult(
                    text=decoded,
                    tokens_in=input_len,
                    tokens_out=tokens_out,
                    input_ids=tokens["input_ids"][row_idx].detach().cpu().tolist(),
                    attention_mask=tokens["attention_mask"][row_idx]
                    .detach()
                    .cpu()
                    .tolist(),
                    generated_ids=generated.detach().cpu().tolist(),
                    effective_alphas=effective_alphas,
                    steering_applied=bool(
                        self.controller
                        and any(
                            abs(value) > 0.0 for value in effective_alphas.values()
                        )
                    ),
                    model_id=self.model_name,
                    model_revision=self.model_revision,
                    tokenizer_revision=self.tokenizer_revision,
                    dtype=self._model_dtype_name(),
                    quantization="nf4" if self.use_quantization else None,
                    do_sample=False,
                    steering_vector_ids=self.steering_vector_ids,
                    steering_vector_hashes=self.steering_vector_hashes,
                )

        if any(res is None for res in results):
            raise RuntimeError("Missing generation results for one or more requests")

        return [cast(GenerationResult, res) for res in results]

    def layers_used(self) -> List[int]:
        if not self.controller:
            return []
        return self.controller.needed_layers


class MockBackend(LanguageBackend):
    def __init__(
        self,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        alpha_strength: float = 1.0,
        per_trait_strength: Optional[Mapping[str, object]] = None,
        suppress_alphas: bool = False,
        do_sample: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            alpha_strength=alpha_strength,
            per_trait_strength=per_trait_strength,
            suppress_alphas=suppress_alphas,
            do_sample=do_sample,
        )
        self.seed = seed

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        alphas: Dict[str, float],
        *,
        sampling_seed: Optional[int] = None,
    ) -> GenerationResult:
        scaled = self._scale_alphas(alphas)
        coeffs = ", ".join(f"{trait}:{alpha:+.2f}" for trait, alpha in sorted(scaled.items()))
        text = (
            f"[mock tokens={max_new_tokens}] Persona[{coeffs}] responds to: "
            f"{prompt.splitlines()[-1]}"
        )
        return GenerationResult(
            text=text,
            tokens_in=len(prompt.split()),
            tokens_out=max_new_tokens,
            effective_alphas=scaled,
            # The mock formats coefficients into text but has no activation
            # hooks, so it must never claim that steering was applied.
            steering_applied=False,
            model_id="mock",
            do_sample=self.do_sample,
            sampling_seed=sampling_seed if self.do_sample else None,
        )

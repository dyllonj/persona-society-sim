"""Language backend abstractions for agents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

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


@dataclass
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int


@dataclass
class BatchGenerationRequest:
    """A single generation request in a batch."""
    prompt: str
    max_new_tokens: int
    alphas: Dict[str, float]


class LanguageBackend:
    def __init__(
        self, temperature: float = 0.7, top_p: float = 0.9, alpha_strength: float = 1.0
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.alpha_strength = alpha_strength

    def _scale_alphas(self, alphas: Dict[str, float]) -> Dict[str, float]:
        if abs(self.alpha_strength - 1.0) < 1e-6:
            return alphas
        return {trait: coeff * self.alpha_strength for trait, coeff in alphas.items()}

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:  # pragma: no cover - interface
        """Generate responses for multiple prompts in parallel."""
        raise NotImplementedError

    def layers_used(self) -> List[int]:  # pragma: no cover - interface
        return []


class HFBackend(LanguageBackend):
    def __init__(
        self,
        model_name: str,
        trait_vectors: Dict[str, Dict[int, torch.Tensor]],
        vector_norms: Optional[Dict[str, Dict[int, float]]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_quantization: bool = False,
        alpha_strength: float = 1.0,
    ):
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError("torch and transformers are required for HFBackend")
        super().__init__(temperature=temperature, top_p=top_p, alpha_strength=alpha_strength)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Reduce Transformers log verbosity (e.g., pad_token warnings)
        if hf_logging is not None:
            try:
                hf_logging.set_verbosity_error()
            except Exception:
                pass

        # Load model with optional quantization
        if use_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

        provided_norms = vector_norms or {}
        self.vector_norms: Dict[str, Dict[int, float]] = {
            trait: dict(per_layer) for trait, per_layer in provided_norms.items()
        }
        self.controller = SteeringController(
            self.model,
            trait_vectors,
            vector_norms=self.vector_norms,
        )
        self.controller.register()

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_length = tokens["input_ids"].shape[-1]
        scaled_alphas = self._scale_alphas(alphas)
        self.controller.set_alphas(scaled_alphas, prompt_length=prompt_length)
        with torch.no_grad():
            try:
                output = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            finally:
                self.controller.clear_prompt_metadata()
        tokens_in = tokens["input_ids"].shape[-1]
        generated = output[0][tokens_in:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        tokens_out = generated.shape[-1]
        return GenerationResult(text=decoded, tokens_in=tokens_in, tokens_out=tokens_out)

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:
        """Generate responses for multiple prompts in parallel using batched inference."""
        if not requests:
            return []

        grouped: Dict[int, List[tuple[int, BatchGenerationRequest]]] = defaultdict(list)
        for idx, req in enumerate(requests):
            grouped[req.max_new_tokens].append((idx, req))

        results: List[Optional[GenerationResult]] = [None] * len(requests)

        for max_tokens, bucket in grouped.items():
            prompts = [req.prompt for _, req in bucket]
            tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            input_lengths = (tokens["attention_mask"].sum(dim=1)).tolist()
            batched_alphas = [self._scale_alphas(req.alphas) for _, req in bucket]
            self.controller.set_batched_alphas(batched_alphas, prompt_lengths=input_lengths)

            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **tokens,
                        max_new_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                finally:
                    self.controller.clear_prompt_metadata()
                    self.controller.clear_batched_alphas()

            for (original_idx, req), output, input_len in zip(bucket, outputs, input_lengths):
                generated = output[input_len : input_len + req.max_new_tokens]
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                tokens_out = generated.shape[-1]
                results[original_idx] = GenerationResult(
                    text=decoded,
                    tokens_in=input_len,
                    tokens_out=tokens_out,
                )

        if any(res is None for res in results):
            raise RuntimeError("Missing generation results for one or more requests")

        return [cast(GenerationResult, res) for res in results]

    def layers_used(self) -> List[int]:
        return self.controller.needed_layers


class MockBackend(LanguageBackend):
    def __init__(
        self,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        alpha_strength: float = 1.0,
    ):
        super().__init__(temperature=temperature, top_p=top_p, alpha_strength=alpha_strength)
        self.seed = seed

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:
        scaled = self._scale_alphas(alphas)
        coeffs = ", ".join(f"{trait}:{alpha:+.2f}" for trait, alpha in sorted(scaled.items()))
        text = (
            f"[mock tokens={max_new_tokens}] Persona[{coeffs}] responds to: "
            f"{prompt.splitlines()[-1]}"
        )
        return GenerationResult(text=text, tokens_in=len(prompt.split()), tokens_out=max_new_tokens)

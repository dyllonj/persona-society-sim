"""Language backend abstractions for agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def __init__(self, temperature: float = 0.7, top_p: float = 0.9):
        self.temperature = temperature
        self.top_p = top_p

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
        layers: List[int],
        trait_vectors: Dict[str, Dict[int, torch.Tensor]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_quantization: bool = False,
    ):
        super().__init__(temperature=temperature, top_p=top_p)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        self.controller = SteeringController(self.model, trait_vectors)
        self.controller.register()
        self._layers = layers

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:
        self.controller.set_alphas(alphas)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        tokens_in = tokens["input_ids"].shape[-1]
        generated = output[0][tokens_in:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        tokens_out = generated.shape[-1]
        return GenerationResult(text=decoded, tokens_in=tokens_in, tokens_out=tokens_out)

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:
        """Generate responses for multiple prompts in parallel using batched inference.

        Note: Currently processes requests with same max_new_tokens together for efficiency.
        If steering vectors differ significantly, we fall back to sequential processing.
        """
        if not requests:
            return []

        # For simplicity, we'll use a common max_new_tokens (the max of all requests)
        # and process all in one batch. More sophisticated implementations could
        # group by max_new_tokens to optimize.
        max_tokens = max(req.max_new_tokens for req in requests)

        # Check if all requests have similar alphas - if yes, we can batch with one steering
        # If not, we need to process sequentially or implement per-sample steering
        first_alphas = requests[0].alphas
        all_same_steering = all(req.alphas == first_alphas for req in requests)

        if not all_same_steering:
            # Fall back to sequential processing with different steering vectors
            return [self.generate(req.prompt, req.max_new_tokens, req.alphas) for req in requests]

        # Batch processing with same steering
        self.controller.set_alphas(first_alphas)

        # Tokenize all prompts with padding
        prompts = [req.prompt for req in requests]
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Track input lengths for each sequence
        input_lengths = (tokens["attention_mask"].sum(dim=1)).tolist()

        with torch.no_grad():
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode each output
        results = []
        for idx, (output, input_len) in enumerate(zip(outputs, input_lengths)):
            generated = output[input_len:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            tokens_out = generated.shape[-1]
            results.append(GenerationResult(
                text=decoded,
                tokens_in=input_len,
                tokens_out=tokens_out,
            ))

        return results

    def layers_used(self) -> List[int]:
        return list(self.controller.needed_layers)


class MockBackend(LanguageBackend):
    def __init__(self, seed: int = 0, temperature: float = 0.0, top_p: float = 1.0):
        super().__init__(temperature=temperature, top_p=top_p)
        self.seed = seed

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:
        coeffs = ", ".join(f"{trait}:{alpha:+.2f}" for trait, alpha in sorted(alphas.items()))
        text = (
            f"[mock tokens={max_new_tokens}] Persona[{coeffs}] responds to: "
            f"{prompt.splitlines()[-1]}"
        )
        return GenerationResult(text=text, tokens_in=len(prompt.split()), tokens_out=max_new_tokens)

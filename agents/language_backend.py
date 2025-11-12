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


class LanguageBackend:
    def __init__(self, temperature: float = 0.7, top_p: float = 0.9):
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:  # pragma: no cover - interface
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
    ):
        super().__init__(temperature=temperature, top_p=top_p)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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

"""Gemini backend for agents using Google GenAI SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from agents.language_backend import LanguageBackend, GenerationResult, BatchGenerationRequest
from steering.prompt_steering import get_steering_prompt


class GeminiBackend(LanguageBackend):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        alpha_strength: float = 1.0,
        suppress_alphas: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            alpha_strength=alpha_strength,
            suppress_alphas=suppress_alphas,
        )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate(self, prompt: str, max_new_tokens: int, alphas: Dict[str, float]) -> GenerationResult:
        """
        Generate text using Gemini.
        Steering is applied by prepending system instructions derived from 'alphas'.
        """
        scaled_alphas = self._scale_alphas(alphas)
        steering_prompt = get_steering_prompt(scaled_alphas)
        
        # Combine steering prompt with user prompt
        # Note: Gemini supports system instructions at model init, but for per-request dynamic steering
        # we prepend it to the prompt or use a chat session. Here we prepend for simplicity in a completion context.
        full_prompt = f"{steering_prompt}\n{prompt}" if steering_prompt else prompt
        
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=max_new_tokens,
        )
        
        # Safety settings - block few things to allow for diverse persona simulation
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            text = response.text
            # Estimate token counts (Gemini API provides usage metadata, but we can also approximate)
            # usage_metadata is available in response.usage_metadata
            tokens_in = 0
            tokens_out = 0
            if response.usage_metadata:
                tokens_in = response.usage_metadata.prompt_token_count
                tokens_out = response.usage_metadata.candidates_token_count
            
            return GenerationResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out)
            
        except Exception as e:
            # Fallback or error handling
            print(f"Gemini generation error: {e}")
            return GenerationResult(text=f"[Error: {e}]", tokens_in=0, tokens_out=0)

    def generate_batch(self, requests: List[BatchGenerationRequest]) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts.
        Gemini API doesn't strictly support batching in the same way as local inference (throughput wise),
        but we can iterate or use async (if we were async). For now, we iterate synchronously.
        """
        results = []
        for req in requests:
            res = self.generate(req.prompt, req.max_new_tokens, req.alphas)
            results.append(res)
        return results

    def layers_used(self) -> List[int]:
        # API models don't expose layers
        return []

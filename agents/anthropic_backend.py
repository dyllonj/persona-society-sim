"""Hosted inference for the existing town; no activation steering or fake replay IDs."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import RLock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agents.language_backend import GenerationResult, LanguageBackend


class AnthropicBackend(LanguageBackend):
    def __init__(self, model_name, *, receipt_dir, temperature=0.7,
                 timeout=60, retries=2, transport=None):
        super().__init__(temperature=temperature, suppress_alphas=True, do_sample=True)
        if not model_name or timeout <= 0 or not 0 <= retries <= 5:
            raise ValueError("Explicit model, positive timeout and 0..5 retries required")
        self.model_name = model_name
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries = timeout, retries
        self._transport = transport or self._post
        self._lock = RLock()

    def _post(self, payload):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for live inference")
        request = Request("https://api.anthropic.com/v1/messages",
                          data=json.dumps(payload).encode(), method="POST",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                   "Content-Type": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries:
                    raise RuntimeError(f"Hosted inference failed (HTTP {exc.code})") from None
            except (URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("Hosted inference timed out or connection failed") from None
            time.sleep(min(2 ** attempt, 8))

    def generate(self, prompt, max_new_tokens, alphas, *, sampling_seed=None):
        if any(value != 0 for value in alphas.values()):
            raise ValueError("Hosted inference cannot apply activation alphas; use prompt_persona")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        payload = {"model": self.model_name, "max_tokens": max_new_tokens,
                   "temperature": self.temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        # Seed identifies a requested replicate for caching; it is NOT sent to the API.
        identity = {"backend": "anthropic-messages-v1", "request": payload,
                    "replicate_seed": sampling_seed}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.receipt_dir / f"{digest}.json"
        with self._lock:
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt["request_sha256"] != digest:
                    raise ValueError("Inference receipt identity mismatch")
                response = receipt["response"]
            else:
                response = self._transport(payload)
                self._validate_response(response)
                receipt = {"schema_version": 1, "request_sha256": digest,
                           "requested_model": self.model_name,
                           "replicate_seed": sampling_seed, "provider_seed_supported": False,
                           "activation_steering": False, "response": response}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(receipt, sort_keys=True))
                temporary.replace(path)
        self._validate_response(response)
        usage = response["usage"]
        return GenerationResult(
            text="".join(block["text"] for block in response["content"] if block["type"] == "text"),
            tokens_in=usage["input_tokens"] + usage.get("cache_read_input_tokens", 0)
                      + usage.get("cache_creation_input_tokens", 0),
            tokens_out=usage["output_tokens"], model_id=response["model"],
            steering_applied=False, do_sample=True, sampling_seed=None)

    @staticmethod
    def _validate_response(response):
        if response.get("stop_reason") != "end_turn":
            raise RuntimeError("Hosted response incomplete or refused; run is not complete")
        if not response.get("model") or not any(
            block.get("type") == "text" and block.get("text")
            for block in response.get("content", [])
        ):
            raise RuntimeError("Hosted response has no model identity or text")
        for key in ("input_tokens", "output_tokens"):
            value = response.get("usage", {}).get(key)
            if type(value) is not int or value < 0:
                raise RuntimeError("Hosted response is missing valid token usage")

    def generate_batch(self, requests):
        return [self.generate(r.prompt, r.max_new_tokens, r.alphas,
                              sampling_seed=r.sampling_seed) for r in requests]

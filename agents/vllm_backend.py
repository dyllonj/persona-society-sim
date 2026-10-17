"""Loopback-only vLLM client with auditable requests and fail-closed transport."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agents.language_backend import GenerationResult, LanguageBackend
from evaluation.personality import strict_json


class VLLMBackend(LanguageBackend):
    def __init__(self, model_name, *, model_revision, receipt_dir,
                 base_url="http://127.0.0.1:8000/v1", temperature=0.2,
                 top_p=0.9, timeout=180, transport=None):
        super().__init__(temperature=temperature, top_p=top_p,
                         suppress_alphas=True, do_sample=temperature > 0)
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Local pilot requires a loopback HTTP endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Local endpoint must not contain credentials, query or fragment")
        if not model_name or not model_revision or timeout <= 0:
            raise ValueError("Model identity, revision and positive timeout required")
        self.base_url = base_url.rstrip("/")
        self.model_name, self.model_revision = model_name, model_revision
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._transport = transport or self._post
        self._lock = RLock()

    def _post(self, payload):
        request = Request(self.base_url + "/chat/completions", method="POST",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return strict_json(response.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f"Local vLLM HTTP {exc.code}; episode incomplete") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Local vLLM connection failed or timed out; episode incomplete") from None

    def generate(self, prompt, max_new_tokens, alphas, *, sampling_seed=None):
        if any(value != 0 for value in alphas.values()):
            raise ValueError("Local vLLM pilot does not implement activation steering")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        seed = None if sampling_seed is None else sampling_seed % (2 ** 32)
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_new_tokens, "temperature": self.temperature,
                   "top_p": self.top_p, "chat_template_kwargs": {"enable_thinking": False}}
        if seed is not None:
            payload["seed"] = seed
        identity = {"backend": "local-vllm-v1", "endpoint": self.base_url,
                    "model_revision": self.model_revision, "request": payload}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.receipt_dir / f"{digest}.json"
        with self._lock:
            if path.exists():
                receipt = strict_json(path.read_text())
                if receipt["request_sha256"] != digest or receipt["identity"] != identity:
                    raise ValueError("Local inference receipt identity mismatch")
                response = receipt["response"]
            else:
                response = self._transport(payload)
                # Store failed/truncated responses too; they must remain inspectable.
                receipt = {"schema_version": 1, "request_sha256": digest,
                           "identity": identity, "response": response}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(receipt, sort_keys=True))
                temporary.replace(path)
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise RuntimeError("Local vLLM response incomplete or refused; inspect receipt")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Local vLLM returned no response text")
        if response.get("model") != self.model_name:
            raise RuntimeError("Local vLLM returned an unexpected model identity")
        usage = response.get("usage", {})
        for key in ("prompt_tokens", "completion_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise RuntimeError("Local vLLM missing valid token usage")
        return GenerationResult(
            text=content, tokens_in=usage["prompt_tokens"], tokens_out=usage["completion_tokens"],
            model_id=response["model"], model_revision=self.model_revision,
            quantization="MLX 4-bit via vLLM Metal", do_sample=self.do_sample,
            sampling_seed=seed, steering_applied=False)

    def generate_batch(self, requests):
        return [self.generate(r.prompt, r.max_new_tokens, r.alphas,
                              sampling_seed=r.sampling_seed) for r in requests]

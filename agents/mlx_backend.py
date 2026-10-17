"""Sequential Apple Silicon inference for local collaboration pilots."""
from threading import RLock

from agents.language_backend import GenerationResult, LanguageBackend


class MLXBackend(LanguageBackend):
    def __init__(self, model_path, *, model_name, model_revision, temperature=0.2,
                 top_p=0.9, max_context_tokens=4096):
        super().__init__(temperature=temperature, top_p=top_p,
                         suppress_alphas=True, do_sample=temperature > 0)
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)
        self.model_name, self.model_revision = model_name, model_revision
        self.max_context_tokens = max_context_tokens
        self._lock = RLock()

    def generate(self, prompt, max_new_tokens, alphas, *, sampling_seed=None):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if any(value != 0 for value in alphas.values()):
            raise ValueError("MLX pilot does not implement activation steering")
        with self._lock:
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            if len(token_ids) + max_new_tokens > self.max_context_tokens:
                raise ValueError("Local context budget exceeded; do not silently truncate observations")
            if sampling_seed is not None:
                mx.random.seed(sampling_seed % (2 ** 32))
            chunks, generated_ids, last = [], [], None
            for last in stream_generate(
                self.model, self.tokenizer, prompt=token_ids, max_tokens=max_new_tokens,
                sampler=make_sampler(temp=self.temperature, top_p=self.top_p),
            ):
                chunks.append(last.text)
                generated_ids.append(last.token)
            if last is None or last.finish_reason == "length":
                raise RuntimeError("Local generation empty or token-limited; run incomplete")
            return GenerationResult(
                text="".join(chunks), tokens_in=last.prompt_tokens,
                tokens_out=last.generation_tokens, input_ids=list(token_ids),
                attention_mask=[1] * len(token_ids), generated_ids=generated_ids,
                model_id=self.model_name, model_revision=self.model_revision,
                tokenizer_revision=self.model_revision, quantization="MLX 4-bit",
                do_sample=self.do_sample, sampling_seed=sampling_seed,
                steering_applied=False)

    def generate_batch(self, requests):
        return [self.generate(r.prompt, r.max_new_tokens, r.alphas,
                              sampling_seed=r.sampling_seed) for r in requests]

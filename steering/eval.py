"""Steering vector evaluation harness for held-out contrast prompts.

This module loads contrast-style JSONL files, compares multiple-choice
accuracy with and without steering vectors, aggregates the deltas, and emits
structured JSON/Markdown reports.  Optional open-ended transcript sampling is
also supported so researchers can manually grade behavioral differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import pvariance
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional heavy dependencies are only needed for the HF harness.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - exercised in tests via fakes
    torch = None  # type: ignore
    transformers = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from data.prompts.schema import load_prompt_items
from steering.hooks import SteeringController
from steering.llm_judge import JudgeClient, JudgeResult, StaticJudgeClient, score_text_with_judge

LOGGER = logging.getLogger(__name__)
OPTION_DELIMITER = "\n"
PRIMARY_LOGPROB_METRIC = "mean_per_continuation_token"

TRAIT_ALIASES: Dict[str, Tuple[str, str]] = {
    "e": ("extraversion", "E"),
    "extraversion": ("extraversion", "E"),
    "a": ("agreeableness", "A"),
    "agreeableness": ("agreeableness", "A"),
    "c": ("conscientiousness", "C"),
    "conscientiousness": ("conscientiousness", "C"),
    "o": ("openness", "O"),
    "openness": ("openness", "O"),
    "n": ("neuroticism", "N"),
    "neuroticism": ("neuroticism", "N"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def report_content_sha256(report: Dict[str, Any]) -> str:
    """Hash scientific report content, excluding self-hash and wall-clock time."""

    unhashed = {
        key: value
        for key, value in report.items()
        if key not in {"generated_at", "report_content_sha256"}
    }
    return _sha256_json(unhashed)


def _coerce_conditional_scores(
    values: Sequence[ConditionalLogprob | float],
) -> List[ConditionalLogprob]:
    """Accept legacy scalar test scorers as one-token mean/sum scores."""

    scores: List[ConditionalLogprob] = []
    for value in values:
        if isinstance(value, ConditionalLogprob):
            scores.append(value)
        else:
            scalar = float(value)
            scores.append(
                ConditionalLogprob(
                    sum_logprob=scalar,
                    mean_logprob=scalar,
                    token_count=1,
                )
            )
    return scores


@dataclass
class PromptRecord:
    """Normalized prompt payload with helpers for evaluation."""

    prompt_id: str
    question: str
    option_a: str
    option_b: str
    option_a_is_high: bool
    option_b_is_high: bool

    @property
    def high_option_index(self) -> int:
        return 0 if self.option_a_is_high else 1

    @property
    def low_option_index(self) -> int:
        return 1 - self.high_option_index


@dataclass
class PromptEvaluation:
    prompt_id: str
    question: str
    option_a: str
    option_b: str
    high_option: str
    baseline_high_logprob: float
    baseline_low_logprob: float
    steered_high_logprob: float
    steered_low_logprob: float
    baseline_correct: bool
    steered_correct: bool
    baseline_high_logprob_sum: float = 0.0
    baseline_low_logprob_sum: float = 0.0
    steered_high_logprob_sum: float = 0.0
    steered_low_logprob_sum: float = 0.0
    high_option_token_count: int = 0
    low_option_token_count: int = 0


@dataclass
class TraitEvaluation:
    trait_name: str
    trait_code: str
    prompt_path: str
    metadata_path: str
    vector_store_id: str
    num_prompts: int
    accuracy_baseline: float
    accuracy_steered: float
    logprob_gap_baseline: float
    logprob_gap_steered: float
    sign_consistency: float
    directional_improvement: float
    prompt_results: List[PromptEvaluation]
    anti_steerable_fraction: float = 0.0
    per_sample_variance: float = 0.0
    summed_logprob_gap_baseline: float = 0.0
    summed_logprob_gap_steered: float = 0.0
    alpha: float = 1.0

    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_steered - self.accuracy_baseline

    @property
    def logprob_gap_delta(self) -> float:
        return self.logprob_gap_steered - self.logprob_gap_baseline

    @property
    def summed_logprob_gap_delta(self) -> float:
        return self.summed_logprob_gap_steered - self.summed_logprob_gap_baseline


@dataclass(frozen=True)
class ConditionalLogprob:
    """Conditional continuation score under both supported normalizations."""

    sum_logprob: float
    mean_logprob: float
    token_count: int


def canonicalize_trait(value: str) -> Tuple[str, str]:
    key = value.strip().lower()
    if key not in TRAIT_ALIASES:
        raise ValueError(f"Unknown trait specifier: {value}")
    return TRAIT_ALIASES[key]


def _prompt_template(record: PromptRecord) -> str:
    return (
        "You will read a short scenario and two contrasting behaviors. "
        "Respond with the behavior you think best reflects the described persona.\n\n"
        f"Scenario: {record.question}\n"
        f"Option A: {record.option_a}\n"
        f"Option B: {record.option_b}\n\n"
        "Answer by repeating the full text of your preferred option."
    )


def _load_prompt_records(path: Path) -> List[PromptRecord]:
    items = load_prompt_items(path)
    records: List[PromptRecord] = []
    for item in items:
        records.append(
            PromptRecord(
                prompt_id=item.id,
                question=item.question_text,
                option_a=item.option_a,
                option_b=item.option_b,
                option_a_is_high=item.option_a_is_high,
                option_b_is_high=item.option_b_is_high,
            )
        )
    return records


class OptionScorer:
    """Protocol-like base class for computing option logprobs."""

    def score_options(
        self,
        prompt_text: str,
        option_texts: Sequence[str],
        *,
        trait_code: Optional[str],
        alpha: float,
    ) -> List[ConditionalLogprob | float]:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_text(
        self,
        prompt_text: str,
        *,
        trait_code: Optional[str],
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface hook
        return None


TorchTensor = torch.Tensor if torch is not None else Any  # type: ignore


class HFContrastScorer(OptionScorer):
    """Hugging Face-backed scorer that toggles steering vectors on demand."""

    def __init__(
        self,
        model_name: str,
        trait_vectors: Dict[str, Dict[int, TorchTensor]],
        *,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        dtype: str = "bf16",
    ) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ModuleNotFoundError(
                "torch and transformers are required for HFContrastScorer"
            )
        tokenizer_revision = tokenizer_revision or model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=tokenizer_revision,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("held-out option scoring requires a fast tokenizer with offsets")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_key = dtype.strip().lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype_key not in dtype_map:
            raise ValueError(
                f"unsupported inference dtype {dtype!r}; expected bf16, fp16, or fp32"
            )
        requested_dtype = dtype_map[dtype_key]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype=requested_dtype,
            device_map="auto",
        )
        self.model.eval()
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None) or model_revision
        )
        self.tokenizer_revision = (
            self.tokenizer.init_kwargs.get("_commit_hash")
            or tokenizer_revision
            or self.model_revision
        )
        try:
            actual_dtype = next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            actual_dtype = requested_dtype
        self.resolved_dtype = str(actual_dtype).removeprefix("torch.")
        self._validate_trait_vectors(trait_vectors)
        self.trait_vectors = trait_vectors
        self.controller: Optional[SteeringController] = None
        if trait_vectors:
            self.controller = SteeringController(self.model, trait_vectors)
            self.controller.register()

    def _validate_trait_vectors(
        self,
        trait_vectors: Dict[str, Dict[int, TorchTensor]],
    ) -> None:
        config = self.model.config
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config() if callable(get_text_config) else config
        hidden_size = int(getattr(text_config, "hidden_size"))
        layer_count = int(getattr(text_config, "num_hidden_layers"))
        for trait, per_layer in trait_vectors.items():
            if not per_layer:
                raise ValueError(f"Trait {trait} has no evaluation vectors")
            for layer, vector in per_layer.items():
                if not 0 <= int(layer) < layer_count:
                    raise ValueError(
                        f"Trait {trait} layer {layer} is outside model range "
                        f"[0, {layer_count - 1}]"
                    )
                if vector.ndim != 1 or int(vector.shape[0]) != hidden_size:
                    raise ValueError(
                        f"Trait {trait} layer {layer} vector shape {tuple(vector.shape)} "
                        f"does not match hidden size {hidden_size}"
                    )

    def _set_alphas(self, trait_code: Optional[str], alpha: float, prompt_length: int) -> None:
        if not self.controller:
            return
        payload = {code: 0.0 for code in self.trait_vectors}
        if trait_code and trait_code in payload:
            payload[trait_code] = alpha
        self.controller.set_alphas(payload, prompt_length=prompt_length)

    def score_options(
        self,
        prompt_text: str,
        option_texts: Sequence[str],
        *,
        trait_code: Optional[str],
        alpha: float,
        prompt_token_length: Optional[int] = None,
    ) -> List[ConditionalLogprob]:
        # The option must be tokenized as part of the exact string scored by the
        # model. Concatenating independently tokenized prompt/option IDs can
        # silently turn `option.` + `I ...` into the unintended `option.I ...`.
        del prompt_token_length  # Retained only for API compatibility.
        scores: List[ConditionalLogprob] = []
        for option in option_texts:
            full_text = f"{prompt_text}{OPTION_DELIMITER}{option}"
            option_char_start = len(prompt_text) + len(OPTION_DELIMITER)
            encoded = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            combined = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            continuation_positions = [
                index
                for index, (_start, end) in enumerate(offsets)
                if int(end) > option_char_start
            ]
            if not continuation_positions:
                scores.append(
                    ConditionalLogprob(
                        sum_logprob=float("-inf"),
                        mean_logprob=float("-inf"),
                        token_count=0,
                    )
                )
                continue
            first_continuation = continuation_positions[0]
            if continuation_positions != list(
                range(first_continuation, len(combined))
            ):
                raise ValueError("option continuation offsets are not contiguous")
            if first_continuation <= 0:
                raise ValueError("option scoring requires at least one prompt token")
            inputs = torch.tensor([combined], device=self.model.device)
            attn = torch.ones_like(inputs)
            if self.controller:
                self._set_alphas(trait_code, alpha, first_continuation)
            try:
                with torch.inference_mode():
                    logits = self.model(input_ids=inputs, attention_mask=attn).logits
            finally:
                if self.controller:
                    self.controller.clear_prompt_metadata()
            log_probs = torch.log_softmax(logits, dim=-1)
            score = 0.0
            for position in continuation_positions:
                prev_idx = position - 1
                token_id = combined[position]
                token_logprob = log_probs[0, prev_idx, token_id]
                score += float(token_logprob.item())
            token_count = len(continuation_positions)
            scores.append(
                ConditionalLogprob(
                    sum_logprob=score,
                    mean_logprob=score / token_count,
                    token_count=token_count,
                )
            )
        return scores

    def generate_text(
        self,
        prompt_text: str,
        *,
        trait_code: Optional[str],
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_len = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        if self.controller:
            self._set_alphas(trait_code, alpha, prompt_len)
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            if self.controller:
                self.controller.clear_prompt_metadata()
        return self.tokenizer.decode(
            generated[0, prompt_len:],
            skip_special_tokens=True,
        )

    def close(self) -> None:
        if self.controller:
            self.controller.remove()


def evaluate_trait_dataset(
    trait_name: str,
    trait_code: str,
    prompt_path: Path,
    prompts: Sequence[PromptRecord],
    scorer: OptionScorer,
    *,
    alpha: float,
    vector_store_id: str,
    metadata_path: Path,
) -> TraitEvaluation:
    prompt_results: List[PromptEvaluation] = []
    baseline_correct = 0
    steered_correct = 0
    baseline_gaps: List[float] = []
    steered_gaps: List[float] = []
    baseline_summed_gaps: List[float] = []
    steered_summed_gaps: List[float] = []
    gap_deltas: List[float] = []
    anti_steerable = 0
    sign_matches = 0
    directional_improvements = 0

    for record in prompts:
        prompt_text = _prompt_template(record)
        baseline_scores = _coerce_conditional_scores(
            scorer.score_options(
                prompt_text,
                [record.option_a, record.option_b],
                trait_code=trait_code,
                alpha=0.0,
            )
        )
        steered_scores = _coerce_conditional_scores(
            scorer.score_options(
                prompt_text,
                [record.option_a, record.option_b],
                trait_code=trait_code,
                alpha=alpha,
            )
        )
        high_idx = record.high_option_index
        low_idx = record.low_option_index

        baseline_gap = (
            baseline_scores[high_idx].mean_logprob
            - baseline_scores[low_idx].mean_logprob
        )
        steered_gap = (
            steered_scores[high_idx].mean_logprob
            - steered_scores[low_idx].mean_logprob
        )
        baseline_summed_gap = (
            baseline_scores[high_idx].sum_logprob
            - baseline_scores[low_idx].sum_logprob
        )
        steered_summed_gap = (
            steered_scores[high_idx].sum_logprob
            - steered_scores[low_idx].sum_logprob
        )
        baseline_gaps.append(baseline_gap)
        steered_gaps.append(steered_gap)
        baseline_summed_gaps.append(baseline_summed_gap)
        steered_summed_gaps.append(steered_summed_gap)
        gap_delta = steered_gap - baseline_gap
        gap_deltas.append(gap_delta)
        if gap_delta < 0:
            anti_steerable += 1

        baseline_choice = (
            0
            if baseline_scores[0].mean_logprob >= baseline_scores[1].mean_logprob
            else 1
        )
        steered_choice = (
            0
            if steered_scores[0].mean_logprob >= steered_scores[1].mean_logprob
            else 1
        )
        baseline_correct += int(baseline_choice == high_idx)
        steered_correct += int(steered_choice == high_idx)

        if baseline_gap == 0 and steered_gap == 0:
            sign_matches += 1
        elif baseline_gap == 0 or steered_gap == 0:
            sign_matches += 0
        elif baseline_gap > 0 and steered_gap > 0:
            sign_matches += 1
        elif baseline_gap < 0 and steered_gap < 0:
            sign_matches += 1

        if steered_gap > baseline_gap:
            directional_improvements += 1

        prompt_results.append(
            PromptEvaluation(
                prompt_id=record.prompt_id,
                question=record.question,
                option_a=record.option_a,
                option_b=record.option_b,
                high_option="A" if high_idx == 0 else "B",
                baseline_high_logprob=baseline_scores[high_idx].mean_logprob,
                baseline_low_logprob=baseline_scores[low_idx].mean_logprob,
                steered_high_logprob=steered_scores[high_idx].mean_logprob,
                steered_low_logprob=steered_scores[low_idx].mean_logprob,
                baseline_correct=baseline_choice == high_idx,
                steered_correct=steered_choice == high_idx,
                baseline_high_logprob_sum=baseline_scores[high_idx].sum_logprob,
                baseline_low_logprob_sum=baseline_scores[low_idx].sum_logprob,
                steered_high_logprob_sum=steered_scores[high_idx].sum_logprob,
                steered_low_logprob_sum=steered_scores[low_idx].sum_logprob,
                high_option_token_count=baseline_scores[high_idx].token_count,
                low_option_token_count=baseline_scores[low_idx].token_count,
            )
        )

    total = max(1, len(prompts))
    evaluation = TraitEvaluation(
        trait_name=trait_name,
        trait_code=trait_code,
        prompt_path=str(prompt_path),
        metadata_path=str(metadata_path),
        vector_store_id=vector_store_id,
        num_prompts=len(prompts),
        accuracy_baseline=baseline_correct / total,
        accuracy_steered=steered_correct / total,
        logprob_gap_baseline=sum(baseline_gaps) / total if baseline_gaps else 0.0,
        logprob_gap_steered=sum(steered_gaps) / total if steered_gaps else 0.0,
        sign_consistency=sign_matches / total,
        directional_improvement=directional_improvements / total,
        prompt_results=prompt_results,
        anti_steerable_fraction=anti_steerable / total,
        per_sample_variance=pvariance(gap_deltas) if len(gap_deltas) > 1 else 0.0,
        summed_logprob_gap_baseline=(
            sum(baseline_summed_gaps) / total if baseline_summed_gaps else 0.0
        ),
        summed_logprob_gap_steered=(
            sum(steered_summed_gaps) / total if steered_summed_gaps else 0.0
        ),
        alpha=alpha,
    )
    return evaluation


def _load_trait_vectors(
    metadata_root: Path,
    traits: Sequence[Tuple[str, str]],
    *,
    model_name: str | None = None,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, dict]]:
    from steering.vector_store import VectorStore  # Local import to avoid heavy deps at import time

    store = VectorStore(metadata_root)
    if torch is None:
        raise ModuleNotFoundError("torch is required to load steering vectors")
    vectors: Dict[str, Dict[int, torch.Tensor]] = {}
    metadata_map: Dict[str, dict] = {}
    for trait_name, trait_code in traits:
        meta_path = metadata_root / f"{trait_code}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata for trait {trait_code}: {meta_path}")
        metadata = json.loads(meta_path.read_text())
        artifact_model = metadata.get("model_name")
        if model_name and artifact_model and str(artifact_model) != model_name:
            raise ValueError(
                f"Trait {trait_code} vector model mismatch: "
                f"artifact={artifact_model!r}, runtime={model_name!r}"
            )
        vector_store_id = metadata.get("vector_store_id") or trait_code
        preferred = metadata.get("preferred_layers") or []
        bundle = store.load(vector_store_id, layers=preferred or None)
        tensor_layers: Dict[int, torch.Tensor] = {}
        resolved_artifacts: Dict[str, dict] = {}
        for layer_id, array in bundle.calibrated_vectors().items():
            tensor_layers[layer_id] = torch.tensor(array, dtype=torch.float32)
            layer_metadata = bundle.layer_metadata.get(layer_id, {})
            vector_path = store.resolve_vector_path(
                layer_metadata.get("vector_path")
                or next(
                    entry.get("vector_path")
                    for entry in metadata.get("layers", [])
                    if int(entry.get("layer_id")) == int(layer_id)
                )
            )
            resolved_artifacts[str(layer_id)] = {
                "path": str(vector_path),
                "sha256": _sha256_file(vector_path),
            }
        vectors[trait_code] = tensor_layers
        metadata = dict(metadata)
        metadata["metadata_sha256"] = _sha256_file(meta_path)
        metadata["resolved_vector_artifacts"] = resolved_artifacts
        metadata_map[trait_code] = metadata
    return vectors, metadata_map


def _resolve_prompt_path(
    trait_name: str,
    prompt_dir: Path,
    prompt_overrides: Dict[str, Path],
    eval_suffix: str,
) -> Path:
    if trait_name in prompt_overrides:
        return prompt_overrides[trait_name]
    candidate = prompt_dir / f"{trait_name}{eval_suffix}.jsonl"
    if candidate.exists():
        return candidate
    fallback = prompt_dir / f"{trait_name}.jsonl"
    if fallback.exists():
        LOGGER.warning(
            "Missing held-out prompts for %s, falling back to training file", trait_name
        )
        return fallback
    raise FileNotFoundError(f"No prompts found for trait {trait_name}")


@dataclass
class TranscriptPrompt:
    prompt_id: str
    prompt: str
    trait_name: Optional[str]


@dataclass
class CoherenceMetrics:
    score: float
    unique_token_ratio: float
    repeated_bigram_fraction: float
    degenerate_token_fraction: float


@dataclass
class GenerationEvaluation:
    prompt_id: str
    trait_name: str
    trait_code: str
    prompt: str
    baseline: str
    steered: str
    baseline_scores: Dict[str, int]
    steered_scores: Dict[str, int]
    trait_expression_delta: float
    baseline_coherence: CoherenceMetrics
    steered_coherence: CoherenceMetrics


def coherence_metrics(text: str) -> CoherenceMetrics:
    """Return lightweight degeneration/coherence signals for generated text."""

    tokens = [token for token in text.lower().split() if token]
    if not tokens:
        return CoherenceMetrics(
            score=0.0,
            unique_token_ratio=0.0,
            repeated_bigram_fraction=1.0,
            degenerate_token_fraction=1.0,
        )

    unique_token_ratio = len(set(tokens)) / len(tokens)
    degenerate = 0
    for previous, current in zip(tokens, tokens[1:]):
        if previous == current:
            degenerate += 1
    degenerate_fraction = degenerate / max(1, len(tokens) - 1)

    bigrams = list(zip(tokens, tokens[1:]))
    if bigrams:
        repeated_bigram_fraction = 1.0 - (len(set(bigrams)) / len(bigrams))
    else:
        repeated_bigram_fraction = 0.0

    score = max(
        0.0,
        min(
            1.0,
            (
                unique_token_ratio
                + (1.0 - repeated_bigram_fraction)
                + (1.0 - degenerate_fraction)
            )
            / 3.0,
        ),
    )
    return CoherenceMetrics(
        score=score,
        unique_token_ratio=unique_token_ratio,
        repeated_bigram_fraction=repeated_bigram_fraction,
        degenerate_token_fraction=degenerate_fraction,
    )


def coherence_score(text: str) -> float:
    """Scalar convenience wrapper for coherence gate checks."""

    return coherence_metrics(text).score


class GenerationEvalScorer:
    """Generate baseline/steered text and score it with a judge client."""

    def __init__(
        self,
        scorer: OptionScorer,
        judge_client: JudgeClient,
        *,
        judge_model: str,
        target_model: str,
        allow_same_family: bool = False,
    ) -> None:
        self.scorer = scorer
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.target_model = target_model
        self.allow_same_family = allow_same_family

    def _judge(self, text: str, traits: Sequence[str]) -> JudgeResult:
        target_model = None if self.allow_same_family else self.target_model
        if self.allow_same_family:
            LOGGER.warning(
                "allow_same_family=True disables judge/target model-family enforcement"
            )
        return score_text_with_judge(
            self.judge_client,
            text,
            judge_model=self.judge_model,
            target_model=target_model,
            traits=traits,
        )

    def evaluate_prompt(
        self,
        prompt: TranscriptPrompt,
        *,
        trait_name: str,
        trait_code: str,
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationEvaluation:
        baseline = self.scorer.generate_text(
            prompt.prompt,
            trait_code=None,
            alpha=0.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        steered = self.scorer.generate_text(
            prompt.prompt,
            trait_code=trait_code,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        baseline_result = self._judge(baseline, [trait_code])
        steered_result = self._judge(steered, [trait_code])
        baseline_score = baseline_result.scores.get(trait_code, 0)
        steered_score = steered_result.scores.get(trait_code, 0)
        return GenerationEvaluation(
            prompt_id=prompt.prompt_id,
            trait_name=trait_name,
            trait_code=trait_code,
            prompt=prompt.prompt,
            baseline=baseline,
            steered=steered,
            baseline_scores=dict(baseline_result.scores),
            steered_scores=dict(steered_result.scores),
            trait_expression_delta=float(steered_score - baseline_score),
            baseline_coherence=coherence_metrics(baseline),
            steered_coherence=coherence_metrics(steered),
        )


def _load_transcript_prompts(path: Path) -> List[TranscriptPrompt]:
    entries: List[TranscriptPrompt] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                prompt_text = payload
                trait_name = None
            else:
                prompt_text = (
                    data.get("prompt")
                    or data.get("question")
                    or data.get("text")
                    or ""
                )
                trait_name = data.get("trait")
            if not prompt_text:
                continue
            entries.append(
                TranscriptPrompt(
                    prompt_id=f"manual-{idx}",
                    prompt=prompt_text,
                    trait_name=trait_name.lower() if trait_name else None,
                )
            )
    return entries


def _collect_transcripts(
    scorer: OptionScorer,
    manual_prompts: Sequence[TranscriptPrompt],
    traits: Sequence[Tuple[str, str]],
    *,
    alpha: float,
    trait_alphas: Optional[Dict[str, float]] = None,
    max_transcripts: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[dict]:
    transcripts: List[dict] = []
    total = 0
    for prompt in manual_prompts:
        target_traits: Iterable[Tuple[str, str]]
        if prompt.trait_name:
            target_traits = [canonicalize_trait(prompt.trait_name)]
        else:
            target_traits = traits
        for trait_name, trait_code in target_traits:
            if max_transcripts is not None and total >= max_transcripts:
                return transcripts
            baseline = scorer.generate_text(
                prompt.prompt,
                trait_code=None,
                alpha=0.0,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            steered = scorer.generate_text(
                prompt.prompt,
                trait_code=trait_code,
                alpha=(trait_alphas or {}).get(trait_code, alpha),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            transcripts.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "trait_name": trait_name,
                    "prompt": prompt.prompt,
                    "baseline": baseline,
                    "steered": steered,
                }
            )
            total += 1
    return transcripts


def parse_alpha_grid(value: Optional[str]) -> List[float]:
    if not value:
        return []
    alphas: List[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        alphas.append(float(stripped))
    if not alphas:
        raise ValueError("--alpha-grid did not contain any numeric alpha values")
    return alphas


def parse_trait_alphas(values: Optional[Sequence[str]]) -> Dict[str, float]:
    """Parse repeatable `trait=value` overrides into canonical trait codes."""

    result: Dict[str, float] = {}
    for raw_value in values or []:
        for item in raw_value.split(","):
            item = item.strip()
            if not item:
                continue
            trait_value, separator, alpha_value = item.partition("=")
            if not separator:
                raise ValueError("Per-trait alphas must use trait=value format")
            _trait_name, trait_code = canonicalize_trait(trait_value)
            if trait_code in result:
                raise ValueError(f"Duplicate per-trait alpha for {trait_code}")
            alpha = float(alpha_value)
            if not math.isfinite(alpha):
                raise ValueError(f"Per-trait alpha for {trait_code} must be finite")
            result[trait_code] = alpha
    return result


def build_archival_provenance(
    *,
    scorer: HFContrastScorer,
    prompt_paths: Dict[str, Path],
    metadata_root: Path,
    metadata_map: Dict[str, dict],
    vector_config: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a hash-complete record for reproducing a held-out evaluation."""

    root = Path(__file__).resolve().parents[1]
    wrapper_path = root / "scripts" / "eval_vectors.sh"
    index_path = metadata_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing vector index: {index_path}")

    vector_files: Dict[str, dict] = {}
    metadata_files: Dict[str, dict] = {}
    for trait_code, metadata in metadata_map.items():
        meta_path = metadata_root / f"{trait_code}.meta.json"
        metadata_files[trait_code] = {
            "path": str(meta_path),
            "sha256": _sha256_file(meta_path),
        }
        for layer, artifact in (
            metadata.get("resolved_vector_artifacts") or {}
        ).items():
            vector_files[f"{trait_code}@{layer}"] = dict(artifact)

    files: Dict[str, Any] = {
        "prompts": {
            trait_name: {"path": str(path), "sha256": _sha256_file(path)}
            for trait_name, path in prompt_paths.items()
        },
        "vector_metadata": metadata_files,
        "vector_index": {
            "path": str(index_path),
            "sha256": _sha256_file(index_path),
        },
        "vectors": vector_files,
        "scripts": {
            "steering_eval": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "eval_vectors_wrapper": {
                "path": str(wrapper_path),
                "sha256": _sha256_file(wrapper_path),
            },
        },
    }
    if vector_config is not None:
        config_path = vector_config.resolve()
        files["vector_config"] = {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        }

    return {
        "scoring": {
            "option_delimiter": OPTION_DELIMITER,
            "tokenization": "combined_prompt_delimiter_continuation_with_offsets",
            "primary_logprob_metric": PRIMARY_LOGPROB_METRIC,
            "secondary_logprob_metric": "sum_over_continuation_tokens",
        },
        "runtime": {
            "python_version": sys.version,
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": getattr(transformers, "__version__", None),
            "dtype": scorer.resolved_dtype,
        },
        "model": {
            "revision": scorer.model_revision,
            "tokenizer_revision": scorer.tokenizer_revision,
            "config_sha256": _sha256_json(scorer.model.config.to_dict()),
        },
        "files": files,
    }


def summarize_generation_evals(evaluations: Sequence[GenerationEvaluation]) -> dict:
    if not evaluations:
        return {
            "count": 0,
            "trait_expression_delta": 0.0,
            "coherence": {"baseline": 0.0, "steered": 0.0},
            "items": [],
        }
    count = len(evaluations)
    return {
        "count": count,
        "trait_expression_delta": sum(
            item.trait_expression_delta for item in evaluations
        )
        / count,
        "coherence": {
            "baseline": sum(item.baseline_coherence.score for item in evaluations)
            / count,
            "steered": sum(item.steered_coherence.score for item in evaluations)
            / count,
        },
        "items": [
            {
                **asdict(item),
                "baseline_coherence": asdict(item.baseline_coherence),
                "steered_coherence": asdict(item.steered_coherence),
            }
            for item in evaluations
        ],
    }


def measure_cross_trait_bleed(
    traits: Sequence[Tuple[str, str]],
    prompt_records: Dict[str, List[PromptRecord]],
    scorer: OptionScorer,
    *,
    alpha: float,
    trait_alphas: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Measure each source trait vector's logprob-gap effect on all trait prompts."""

    matrix: Dict[str, Dict[str, float]] = {}
    for source_name, source_code in traits:
        source_alpha = (trait_alphas or {}).get(source_code, alpha)
        row: Dict[str, float] = {}
        for target_name, _target_code in traits:
            deltas: List[float] = []
            for record in prompt_records.get(target_name, []):
                prompt_text = _prompt_template(record)
                baseline_scores = _coerce_conditional_scores(
                    scorer.score_options(
                        prompt_text,
                        [record.option_a, record.option_b],
                        trait_code=source_code,
                        alpha=0.0,
                    )
                )
                steered_scores = _coerce_conditional_scores(
                    scorer.score_options(
                        prompt_text,
                        [record.option_a, record.option_b],
                        trait_code=source_code,
                        alpha=source_alpha,
                    )
                )
                high_idx = record.high_option_index
                low_idx = record.low_option_index
                baseline_gap = (
                    baseline_scores[high_idx].mean_logprob
                    - baseline_scores[low_idx].mean_logprob
                )
                steered_gap = (
                    steered_scores[high_idx].mean_logprob
                    - steered_scores[low_idx].mean_logprob
                )
                deltas.append(float(steered_gap - baseline_gap))
            row[target_name] = sum(deltas) / len(deltas) if deltas else 0.0
        matrix[source_name] = row
    return matrix


def _build_markdown(report: dict) -> str:
    lines = ["# Steering Vector Evaluation", ""]
    lines.append(f"*Model*: {report['model']}")
    if report.get("resolved_dtype"):
        lines.append(f"*Resolved dtype*: {report['resolved_dtype']}")
    lines.append(f"*Alpha*: {report['alpha']}")
    if report.get("trait_alphas"):
        lines.append(f"*Per-trait alphas*: {report['trait_alphas']}")
    lines.append(f"*Primary logprob metric*: {PRIMARY_LOGPROB_METRIC}")
    lines.append(f"*Timestamp*: {report['generated_at']}")
    lines.append("")
    lines.append(
        "| Trait | Alpha | Prompts | Baseline Acc. | Steered Acc. | Delta Acc. | "
        "Mean logprob gap Δ | Summed logprob gap Δ | Sign Consistency | Anti-Steerable |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for trait in report.get("traits", []):
        lines.append(
            "| {name} ({code}) | {alpha:.3f} | {count} | {base:.3f} | {steered:.3f} | "
            "{delta:.3f} | {mean_gap:.3f} | {sum_gap:.3f} | {sign:.3f} | {anti:.3f} |".format(
                name=trait["trait_name"],
                code=trait["trait_code"],
                alpha=trait.get("alpha", report["alpha"]),
                count=trait["num_prompts"],
                base=trait["accuracy"]["baseline"],
                steered=trait["accuracy"]["steered"],
                delta=trait["accuracy"]["delta"],
                mean_gap=trait["logprob_gap"]["delta"],
                sum_gap=trait["summed_logprob_gap"]["delta"],
                sign=trait["sign_consistency"],
                anti=trait.get("anti_steerable_fraction", 0.0),
            )
        )
    lines.append("")
    thresholds = report.get("thresholds") or {}
    if thresholds:
        lines.append("## Thresholds")
        for key, value in thresholds.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if report.get("grading_prompts"):
        lines.append("## Grading prompts")
        if report["grading_prompts"].get("gpt4"):
            lines.append(f"- GPT-4: {report['grading_prompts']['gpt4']}")
        if report["grading_prompts"].get("claude"):
            lines.append(f"- Claude: {report['grading_prompts']['claude']}")
        lines.append("")
    if report.get("manual_transcripts"):
        lines.append("## Manual transcripts")
        for entry in report["manual_transcripts"]:
            lines.append(f"### {entry['trait_name']} — {entry['prompt_id']}")
            lines.append(f"Prompt: {entry['prompt']}")
            lines.append("- Baseline: " + entry["baseline"].strip())
            lines.append("- Steered: " + entry["steered"].strip())
            lines.append("")
    generation_eval = report.get("generation_eval") or {}
    if generation_eval.get("count"):
        lines.append("## Generation eval")
        lines.append(
            "- Trait expression delta: {delta:.3f}".format(
                delta=generation_eval.get("trait_expression_delta", 0.0)
            )
        )
        coherence = generation_eval.get("coherence") or {}
        lines.append(
            "- Coherence baseline/steered: {base:.3f}/{steered:.3f}".format(
                base=coherence.get("baseline", 0.0),
                steered=coherence.get("steered", 0.0),
            )
        )
        lines.append("")
    if report.get("alpha_grid"):
        lines.append("## Alpha grid")
        lines.append("| Alpha | Trait | Delta Acc. | Logprob Delta | Anti-Steerable |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in report["alpha_grid"]:
            for trait in row.get("traits", []):
                lines.append(
                    "| {alpha:.3f} | {trait} | {acc:.3f} | {gap:.3f} | {anti:.3f} |".format(
                        alpha=row["alpha"],
                        trait=f"{trait['trait_name']} ({trait['trait_code']})",
                        acc=trait["accuracy"]["delta"],
                        gap=trait["logprob_gap"]["delta"],
                        anti=trait.get("anti_steerable_fraction", 0.0),
                    )
                )
        lines.append("")
    if report.get("bleed_matrix"):
        lines.append("## Cross-trait bleed")
        traits = sorted(report["bleed_matrix"])
        header = "| Source \\ Target | " + " | ".join(traits) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---" for _ in traits) + " |")
        for source in traits:
            row = report["bleed_matrix"].get(source, {})
            values = " | ".join(f"{row.get(target, 0.0):.3f}" for target in traits)
            lines.append(f"| {source} | {values} |")
        lines.append("")
    if report.get("failures"):
        lines.append("## Failing conditions")
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def _trait_to_dict(evaluation: TraitEvaluation) -> dict:
    return {
        "trait_name": evaluation.trait_name,
        "trait_code": evaluation.trait_code,
        "prompt_path": evaluation.prompt_path,
        "metadata_path": evaluation.metadata_path,
        "vector_store_id": evaluation.vector_store_id,
        "alpha": evaluation.alpha,
        "num_prompts": evaluation.num_prompts,
        "accuracy": {
            "baseline": evaluation.accuracy_baseline,
            "steered": evaluation.accuracy_steered,
            "delta": evaluation.accuracy_delta,
        },
        "logprob_gap": {
            "metric": PRIMARY_LOGPROB_METRIC,
            "baseline": evaluation.logprob_gap_baseline,
            "steered": evaluation.logprob_gap_steered,
            "delta": evaluation.logprob_gap_delta,
        },
        "summed_logprob_gap": {
            "metric": "sum_over_continuation_tokens",
            "baseline": evaluation.summed_logprob_gap_baseline,
            "steered": evaluation.summed_logprob_gap_steered,
            "delta": evaluation.summed_logprob_gap_delta,
        },
        "sign_consistency": evaluation.sign_consistency,
        "directional_improvement": evaluation.directional_improvement,
        "anti_steerable_fraction": evaluation.anti_steerable_fraction,
        "per_sample_variance": evaluation.per_sample_variance,
        "per_prompt": [asdict(item) for item in evaluation.prompt_results],
    }


def build_report(
    *,
    model_name: str,
    alpha: float,
    traits: Sequence[TraitEvaluation],
    metadata: Dict[str, dict],
    transcripts: Sequence[dict],
    grading_prompts: Dict[str, Optional[str]],
    thresholds: Dict[str, float],
    generation_evals: Sequence[GenerationEvaluation] = (),
    alpha_grid_results: Sequence[dict] = (),
    bleed_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    trait_alphas: Optional[Dict[str, float]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> dict:
    report = {
        "model": model_name,
        "alpha": alpha,
        "trait_alphas": trait_alphas or {},
        "primary_logprob_metric": PRIMARY_LOGPROB_METRIC,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "traits": [_trait_to_dict(trait) for trait in traits],
        "metadata": metadata,
        "manual_transcripts": list(transcripts),
        "grading_prompts": grading_prompts,
        "thresholds": thresholds,
        "generation_eval": summarize_generation_evals(generation_evals),
        "alpha_grid": list(alpha_grid_results),
        "bleed_matrix": bleed_matrix or {},
        "provenance": provenance or {},
    }
    return report


def _summarize_failures(
    traits: Sequence[TraitEvaluation],
    delta_threshold: Optional[float],
    sign_threshold: Optional[float],
    anti_steerable_threshold: Optional[float] = None,
) -> List[str]:
    failures: List[str] = []
    for trait in traits:
        if delta_threshold is not None and trait.accuracy_delta < delta_threshold:
            saturated_accuracy = (
                trait.accuracy_baseline >= 1.0
                and trait.accuracy_steered >= trait.accuracy_baseline
                and trait.logprob_gap_delta >= 0.0
            )
            if not saturated_accuracy:
                failures.append(
                    f"{trait.trait_name} delta {trait.accuracy_delta:.3f} < {delta_threshold:.3f}"
                )
        if sign_threshold is not None and trait.sign_consistency < sign_threshold:
            failures.append(
                f"{trait.trait_name} sign {trait.sign_consistency:.3f} < {sign_threshold:.3f}"
            )
        if (
            anti_steerable_threshold is not None
            and trait.anti_steerable_fraction > anti_steerable_threshold
        ):
            failures.append(
                f"{trait.trait_name} anti-steerable {trait.anti_steerable_fraction:.3f} > "
                f"{anti_steerable_threshold:.3f}"
            )
    return failures


def _parse_prompt_overrides(values: Optional[Sequence[str]]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("Prompt overrides must use trait=path format")
        trait_key, path_value = value.split("=", 1)
        trait_name, _ = canonicalize_trait(trait_key)
        overrides[trait_name] = Path(path_value)
    return overrides


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate steering vectors.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--traits", nargs="*", default=["extraversion", "agreeableness", "conscientiousness"])
    parser.add_argument("--metadata-root", type=Path, default=Path("data/vectors"))
    parser.add_argument("--vector-config", type=Path)
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--prompt-override", action="append", dest="prompt_overrides")
    parser.add_argument("--eval-suffix", default="_eval")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--trait-alpha",
        action="append",
        default=[],
        help="Repeatable trait=value override; comma-separated entries are also accepted",
    )
    parser.add_argument("--alpha-grid", help="Comma-separated alpha values for dose-response eval")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
        default="bf16",
    )
    parser.add_argument("--delta-threshold", type=float)
    parser.add_argument("--sign-threshold", type=float)
    parser.add_argument("--anti-steerable-threshold", type=float)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--transcript-prompts", type=Path)
    parser.add_argument("--max-transcripts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--generation-eval", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-static-output", action="append")
    parser.add_argument("--allow-same-family", action="store_true")
    parser.add_argument("--measure-bleed", action="store_true")
    parser.add_argument("--gpt4-prompt")
    parser.add_argument("--claude-prompt")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    trait_specs = [canonicalize_trait(value) for value in args.traits]
    trait_alphas = parse_trait_alphas(args.trait_alpha)
    selected_trait_codes = {trait_code for _trait_name, trait_code in trait_specs}
    unknown_alpha_traits = set(trait_alphas) - selected_trait_codes
    if unknown_alpha_traits:
        raise ValueError(
            f"Per-trait alpha overrides were supplied for unselected traits: "
            f"{sorted(unknown_alpha_traits)}"
        )
    prompt_overrides = _parse_prompt_overrides(args.prompt_overrides)
    prompt_records: Dict[str, List[PromptRecord]] = {}
    prompt_paths: Dict[str, Path] = {}
    for trait_name, _ in trait_specs:
        prompt_path = _resolve_prompt_path(
            trait_name, args.prompt_dir, prompt_overrides, args.eval_suffix
        )
        prompt_records[trait_name] = _load_prompt_records(prompt_path)
        prompt_paths[trait_name] = prompt_path

    vectors, metadata_map = _load_trait_vectors(
        args.metadata_root,
        trait_specs,
        model_name=args.model,
    )
    scorer = HFContrastScorer(
        args.model,
        vectors,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        dtype=args.dtype,
    )

    def evaluate_all(
        alpha_value: float,
        *,
        overrides: Optional[Dict[str, float]] = None,
    ) -> List[TraitEvaluation]:
        rows: List[TraitEvaluation] = []
        for trait_name, trait_code in trait_specs:
            prompts = prompt_records.get(trait_name, [])
            meta_path = args.metadata_root / f"{trait_code}.meta.json"
            metadata = metadata_map.get(trait_code, {})
            vector_store_id = metadata.get("vector_store_id") or trait_code
            evaluation = evaluate_trait_dataset(
                trait_name,
                trait_code,
                prompt_paths[trait_name],
                prompts,
                scorer,
                alpha=(overrides or {}).get(trait_code, alpha_value),
                vector_store_id=vector_store_id,
                metadata_path=meta_path,
            )
            rows.append(evaluation)
        return rows

    evaluations = evaluate_all(args.alpha, overrides=trait_alphas)
    alpha_grid_results: List[dict] = []
    for alpha_value in parse_alpha_grid(args.alpha_grid):
        alpha_grid_results.append(
            {
                "alpha": alpha_value,
                "traits": [_trait_to_dict(row) for row in evaluate_all(alpha_value)],
            }
        )

    manual_transcripts: List[dict] = []
    if args.transcript_prompts:
        manual_prompts = _load_transcript_prompts(args.transcript_prompts)
        manual_transcripts = _collect_transcripts(
            scorer,
            manual_prompts,
            trait_specs,
            alpha=args.alpha,
            trait_alphas=trait_alphas,
            max_transcripts=args.max_transcripts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    generation_evals: List[GenerationEvaluation] = []
    if args.generation_eval:
        if not args.transcript_prompts:
            raise ValueError("--generation-eval requires --transcript-prompts")
        if not args.judge_model:
            raise ValueError("--generation-eval requires --judge-model")
        if not args.judge_static_output:
            raise ValueError(
                "--generation-eval currently requires --judge-static-output for a "
                "mockable JudgeClient; provider clients can implement JudgeClient."
            )
        manual_prompts = _load_transcript_prompts(args.transcript_prompts)
        judge_client = StaticJudgeClient(args.judge_static_output)
        generation_scorer = GenerationEvalScorer(
            scorer,
            judge_client,
            judge_model=args.judge_model,
            target_model=args.model,
            allow_same_family=args.allow_same_family,
        )
        for prompt in manual_prompts:
            target_traits: Iterable[Tuple[str, str]]
            if prompt.trait_name:
                target_traits = [canonicalize_trait(prompt.trait_name)]
            else:
                target_traits = trait_specs
            for trait_name, trait_code in target_traits:
                if args.max_transcripts is not None and len(generation_evals) >= args.max_transcripts:
                    break
                generation_evals.append(
                    generation_scorer.evaluate_prompt(
                        prompt,
                        trait_name=trait_name,
                        trait_code=trait_code,
                        alpha=trait_alphas.get(trait_code, args.alpha),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                )

    bleed_matrix = (
        measure_cross_trait_bleed(
            trait_specs,
            prompt_records,
            scorer,
            alpha=args.alpha,
            trait_alphas=trait_alphas,
        )
        if args.measure_bleed
        else {}
    )

    scorer.close()

    thresholds = {
        key: value
        for key, value in {
            "delta_threshold": args.delta_threshold,
            "sign_threshold": args.sign_threshold,
            "anti_steerable_threshold": args.anti_steerable_threshold,
        }.items()
        if value is not None
    }

    grading_prompts = {"gpt4": args.gpt4_prompt, "claude": args.claude_prompt}

    report = build_report(
        model_name=args.model,
        alpha=args.alpha,
        traits=evaluations,
        metadata=metadata_map,
        transcripts=manual_transcripts,
        grading_prompts=grading_prompts,
        thresholds=thresholds,
        generation_evals=generation_evals,
        alpha_grid_results=alpha_grid_results,
        bleed_matrix=bleed_matrix,
        trait_alphas={
            trait_code: trait_alphas.get(trait_code, args.alpha)
            for _trait_name, trait_code in trait_specs
        },
        provenance=build_archival_provenance(
            scorer=scorer,
            prompt_paths=prompt_paths,
            metadata_root=args.metadata_root,
            metadata_map=metadata_map,
            vector_config=args.vector_config,
        ),
    )
    report["model_revision"] = scorer.model_revision
    report["tokenizer_revision"] = scorer.tokenizer_revision
    report["resolved_dtype"] = scorer.resolved_dtype

    failures = _summarize_failures(
        evaluations,
        args.delta_threshold,
        args.sign_threshold,
        args.anti_steerable_threshold,
    )
    report["failures"] = failures
    report["report_content_sha256"] = report_content_sha256(report)

    json_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_payload)
    else:
        print(json_payload)

    markdown = _build_markdown(report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    else:
        print(markdown)

    return 1 if failures else 0


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

"""Steering vector evaluation harness for held-out contrast prompts.

This module loads contrast-style JSONL files, compares multiple-choice
accuracy with and without steering vectors, aggregates the deltas, and emits
structured JSON/Markdown reports.  Optional open-ended transcript sampling is
also supported so researchers can manually grade behavioral differences.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional heavy dependencies are only needed for the HF harness.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - exercised in tests via fakes
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from data.prompts.schema import load_prompt_items
from steering.hooks import SteeringController

LOGGER = logging.getLogger(__name__)

TRAIT_ALIASES: Dict[str, Tuple[str, str]] = {
    "e": ("extraversion", "E"),
    "extraversion": ("extraversion", "E"),
    "a": ("agreeableness", "A"),
    "agreeableness": ("agreeableness", "A"),
    "c": ("conscientiousness", "C"),
    "conscientiousness": ("conscientiousness", "C"),
}


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

    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_steered - self.accuracy_baseline

    @property
    def logprob_gap_delta(self) -> float:
        return self.logprob_gap_steered - self.logprob_gap_baseline


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
    ) -> List[float]:  # pragma: no cover - interface
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
    ) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ModuleNotFoundError(
                "torch and transformers are required for HFContrastScorer"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if hasattr(torch, "float16") else None,
            device_map="auto",
        )
        self.model.eval()
        self.trait_vectors = trait_vectors
        self.controller: Optional[SteeringController] = None
        if trait_vectors:
            self.controller = SteeringController(self.model, trait_vectors)
            self.controller.register()

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
    ) -> List[float]:
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        if self.controller:
            self._set_alphas(trait_code, alpha, prompt_len)
        scores: List[float] = []
        for option in option_texts:
            option_ids = self.tokenizer(option, add_special_tokens=False)["input_ids"]
            if not option_ids:
                scores.append(float("-inf"))
                continue
            combined = prompt_ids + option_ids
            inputs = torch.tensor([combined], device=self.model.device)
            attn = torch.ones_like(inputs)
            with torch.no_grad():
                logits = self.model(input_ids=inputs, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            score = 0.0
            for position in range(prompt_len, len(combined)):
                prev_idx = position - 1
                if prev_idx < 0:
                    continue
                token_id = combined[position]
                token_logprob = log_probs[0, prev_idx, token_id]
                score += float(token_logprob.item())
            scores.append(score)
        if self.controller:
            self.controller.clear_prompt_metadata()
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
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        if self.controller:
            self.controller.clear_prompt_metadata()
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

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
    sign_matches = 0
    directional_improvements = 0

    for record in prompts:
        prompt_text = _prompt_template(record)
        baseline_scores = scorer.score_options(
            prompt_text,
            [record.option_a, record.option_b],
            trait_code=trait_code,
            alpha=0.0,
        )
        steered_scores = scorer.score_options(
            prompt_text,
            [record.option_a, record.option_b],
            trait_code=trait_code,
            alpha=alpha,
        )
        high_idx = record.high_option_index
        low_idx = record.low_option_index

        baseline_gap = baseline_scores[high_idx] - baseline_scores[low_idx]
        steered_gap = steered_scores[high_idx] - steered_scores[low_idx]
        baseline_gaps.append(baseline_gap)
        steered_gaps.append(steered_gap)

        baseline_choice = 0 if baseline_scores[0] >= baseline_scores[1] else 1
        steered_choice = 0 if steered_scores[0] >= steered_scores[1] else 1
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
                baseline_high_logprob=float(baseline_scores[high_idx]),
                baseline_low_logprob=float(baseline_scores[low_idx]),
                steered_high_logprob=float(steered_scores[high_idx]),
                steered_low_logprob=float(steered_scores[low_idx]),
                baseline_correct=baseline_choice == high_idx,
                steered_correct=steered_choice == high_idx,
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
    )
    return evaluation


def _load_trait_vectors(
    metadata_root: Path, traits: Sequence[Tuple[str, str]]
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
        vector_store_id = metadata.get("vector_store_id") or trait_code
        preferred = metadata.get("preferred_layers") or []
        bundle = store.load(vector_store_id, layers=preferred or None)
        tensor_layers: Dict[int, torch.Tensor] = {}
        for layer_id, array in bundle.vectors.items():
            tensor_layers[layer_id] = torch.tensor(array, dtype=torch.float32)
        vectors[trait_code] = tensor_layers
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
                alpha=alpha,
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


def _build_markdown(report: dict) -> str:
    lines = ["# Steering Vector Evaluation", ""]
    lines.append(f"*Model*: {report['model']}")
    lines.append(f"*Alpha*: {report['alpha']}")
    lines.append(f"*Timestamp*: {report['generated_at']}")
    lines.append("")
    lines.append(
        "| Trait | Prompts | Baseline Acc. | Steered Acc. | Δ Acc. | Sign Consistency |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for trait in report.get("traits", []):
        lines.append(
            "| {name} ({code}) | {count} | {base:.3f} | {steered:.3f} | {delta:.3f} | {sign:.3f} |".format(
                name=trait["trait_name"],
                code=trait["trait_code"],
                count=trait["num_prompts"],
                base=trait["accuracy"]["baseline"],
                steered=trait["accuracy"]["steered"],
                delta=trait["accuracy"]["delta"],
                sign=trait["sign_consistency"],
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
        "num_prompts": evaluation.num_prompts,
        "accuracy": {
            "baseline": evaluation.accuracy_baseline,
            "steered": evaluation.accuracy_steered,
            "delta": evaluation.accuracy_delta,
        },
        "logprob_gap": {
            "baseline": evaluation.logprob_gap_baseline,
            "steered": evaluation.logprob_gap_steered,
            "delta": evaluation.logprob_gap_delta,
        },
        "sign_consistency": evaluation.sign_consistency,
        "directional_improvement": evaluation.directional_improvement,
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
) -> dict:
    report = {
        "model": model_name,
        "alpha": alpha,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "traits": [_trait_to_dict(trait) for trait in traits],
        "metadata": metadata,
        "manual_transcripts": list(transcripts),
        "grading_prompts": grading_prompts,
        "thresholds": thresholds,
    }
    return report


def _summarize_failures(
    traits: Sequence[TraitEvaluation],
    delta_threshold: Optional[float],
    sign_threshold: Optional[float],
) -> List[str]:
    failures: List[str] = []
    for trait in traits:
        if delta_threshold is not None and trait.accuracy_delta < delta_threshold:
            failures.append(
                f"{trait.trait_name} delta {trait.accuracy_delta:.3f} < {delta_threshold:.3f}"
            )
        if sign_threshold is not None and trait.sign_consistency < sign_threshold:
            failures.append(
                f"{trait.trait_name} sign {trait.sign_consistency:.3f} < {sign_threshold:.3f}"
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
    parser.add_argument("--traits", nargs="*", default=["extraversion", "agreeableness", "conscientiousness"])
    parser.add_argument("--metadata-root", type=Path, default=Path("data/vectors"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--prompt-override", action="append", dest="prompt_overrides")
    parser.add_argument("--eval-suffix", default="_eval")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--delta-threshold", type=float)
    parser.add_argument("--sign-threshold", type=float)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--transcript-prompts", type=Path)
    parser.add_argument("--max-transcripts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpt4-prompt")
    parser.add_argument("--claude-prompt")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    trait_specs = [canonicalize_trait(value) for value in args.traits]
    prompt_overrides = _parse_prompt_overrides(args.prompt_overrides)
    prompt_records: Dict[str, List[PromptRecord]] = {}
    prompt_paths: Dict[str, Path] = {}
    for trait_name, _ in trait_specs:
        prompt_path = _resolve_prompt_path(
            trait_name, args.prompt_dir, prompt_overrides, args.eval_suffix
        )
        prompt_records[trait_name] = _load_prompt_records(prompt_path)
        prompt_paths[trait_name] = prompt_path

    vectors, metadata_map = _load_trait_vectors(args.metadata_root, trait_specs)
    scorer = HFContrastScorer(args.model, vectors)

    evaluations: List[TraitEvaluation] = []
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
            alpha=args.alpha,
            vector_store_id=vector_store_id,
            metadata_path=meta_path,
        )
        evaluations.append(evaluation)

    manual_transcripts: List[dict] = []
    if args.transcript_prompts:
        manual_prompts = _load_transcript_prompts(args.transcript_prompts)
        manual_transcripts = _collect_transcripts(
            scorer,
            manual_prompts,
            trait_specs,
            alpha=args.alpha,
            max_transcripts=args.max_transcripts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    scorer.close()

    thresholds = {
        key: value
        for key, value in {
            "delta_threshold": args.delta_threshold,
            "sign_threshold": args.sign_threshold,
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
    )

    failures = _summarize_failures(
        evaluations, args.delta_threshold, args.sign_threshold
    )
    report["failures"] = failures

    json_payload = json.dumps(report, indent=2)
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

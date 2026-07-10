from pathlib import Path

import torch
import pytest

from steering import eval as steering_eval
from steering.llm_judge import StaticJudgeClient


class FakeScorer(steering_eval.OptionScorer):
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.calls = []

    def score_options(self, prompt_text, option_texts, *, trait_code, alpha):
        self.calls.append((trait_code, alpha, prompt_text))
        return self.sequences.pop(0)

    def generate_text(self, prompt_text, *, trait_code, alpha, max_new_tokens, temperature, top_p):
        return f"{prompt_text}::{trait_code or 'none'}::{alpha}"


class _RecordingController:
    def __init__(self):
        self.set_calls = []
        self.clear_calls = 0

    def set_alphas(self, payload, prompt_length=None):
        self.set_calls.append((dict(payload), prompt_length))

    def clear_prompt_metadata(self):
        self.clear_calls += 1


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [1, 2] if text == "prompt" else [3]}


class _TinyModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        logits = torch.zeros((1, input_ids.shape[1], 8), dtype=torch.float32)
        return type("Output", (), {"logits": logits})()


def test_hf_option_scorer_resets_prompt_mask_for_every_option():
    scorer = object.__new__(steering_eval.HFContrastScorer)
    scorer.tokenizer = _TinyTokenizer()
    scorer.model = _TinyModel()
    scorer.trait_vectors = {"E": {0: torch.ones(1)}}
    scorer.controller = _RecordingController()

    scores = scorer.score_options(
        "prompt",
        ["first", "second"],
        trait_code="E",
        alpha=0.75,
    )

    assert len(scores) == 2
    assert scorer.controller.set_calls == [({"E": 0.75}, 2), ({"E": 0.75}, 2)]
    assert scorer.controller.clear_calls == 2


def test_evaluate_trait_dataset_computes_metrics():
    prompt_path = Path("tests/data/prompts/synthetic_trait.jsonl")
    prompts = steering_eval._load_prompt_records(prompt_path)
    scorer = FakeScorer(
        [
            [-0.2, -0.5],  # prompt 1 baseline (correct)
            [-0.1, -0.8],  # prompt 1 steered (correct)
            [-0.2, -0.4],  # prompt 2 baseline (incorrect)
            [-0.6, -0.1],  # prompt 2 steered (correct)
        ]
    )
    result = steering_eval.evaluate_trait_dataset(
        "extraversion",
        "E",
        prompt_path,
        prompts,
        scorer,
        alpha=1.0,
        vector_store_id="E",
        metadata_path=Path("vectors/E.meta.json"),
    )

    assert result.num_prompts == 2
    assert result.accuracy_baseline == 0.5
    assert result.accuracy_steered == 1.0
    assert result.accuracy_delta == 0.5
    assert result.sign_consistency == 0.5
    assert result.anti_steerable_fraction == 0.0
    assert result.per_sample_variance > 0.0
    assert len(result.prompt_results) == 2
    assert result.prompt_results[1].high_option == "B"


def test_canonicalize_trait_handles_aliases():
    assert steering_eval.canonicalize_trait("E") == ("extraversion", "E")
    assert steering_eval.canonicalize_trait("agreeableness") == ("agreeableness", "A")


def test_vector_loader_rejects_artifacts_from_another_model():
    with pytest.raises(ValueError, match="vector model mismatch"):
        steering_eval._load_trait_vectors(
            Path("data/vectors"),
            [("extraversion", "E")],
            model_name="wrong/model",
        )


def test_build_markdown_includes_failures():
    eval_entry = steering_eval.TraitEvaluation(
        trait_name="extraversion",
        trait_code="E",
        prompt_path="prompts.jsonl",
        metadata_path="vectors/E.meta.json",
        vector_store_id="E",
        num_prompts=2,
        accuracy_baseline=0.4,
        accuracy_steered=0.9,
        logprob_gap_baseline=0.1,
        logprob_gap_steered=0.5,
        sign_consistency=0.5,
        directional_improvement=1.0,
        prompt_results=[],
    )
    report = steering_eval.build_report(
        model_name="mock-model",
        alpha=1.0,
        traits=[eval_entry],
        metadata={},
        transcripts=[],
        grading_prompts={"gpt4": None, "claude": None},
        thresholds={"delta": 0.2},
    )
    report["failures"] = ["extraversion delta 0.1 < 0.2"]
    markdown = steering_eval._build_markdown(report)
    assert "extraversion (E)" in markdown
    assert "extraversion delta 0.1 < 0.2" in markdown


def test_saturated_accuracy_gate_uses_logprob_direction():
    saturated = steering_eval.TraitEvaluation(
        trait_name="extraversion",
        trait_code="E",
        prompt_path="prompts.jsonl",
        metadata_path="vectors/E.meta.json",
        vector_store_id="E",
        num_prompts=2,
        accuracy_baseline=1.0,
        accuracy_steered=1.0,
        logprob_gap_baseline=0.1,
        logprob_gap_steered=0.2,
        sign_consistency=1.0,
        directional_improvement=1.0,
        prompt_results=[],
    )
    failures = steering_eval._summarize_failures(
        [saturated],
        delta_threshold=0.1,
        sign_threshold=None,
        anti_steerable_threshold=None,
    )

    assert failures == []


def test_saturated_accuracy_gate_fails_when_logprob_regresses():
    regressed = steering_eval.TraitEvaluation(
        trait_name="extraversion",
        trait_code="E",
        prompt_path="prompts.jsonl",
        metadata_path="vectors/E.meta.json",
        vector_store_id="E",
        num_prompts=2,
        accuracy_baseline=1.0,
        accuracy_steered=1.0,
        logprob_gap_baseline=0.2,
        logprob_gap_steered=0.1,
        sign_consistency=1.0,
        directional_improvement=0.0,
        prompt_results=[],
    )
    failures = steering_eval._summarize_failures(
        [regressed],
        delta_threshold=0.1,
        sign_threshold=None,
        anti_steerable_threshold=None,
    )

    assert failures == ["extraversion delta 0.000 < 0.100"]


def test_coherence_score_penalizes_repetition():
    coherent = steering_eval.coherence_score("I will organize the notes and share them.")
    repetitive = steering_eval.coherence_score("yes yes yes yes")

    assert coherent > repetitive


def test_generation_eval_scorer_scores_trait_delta():
    scorer = FakeScorer([])
    judge = StaticJudgeClient(['{"E": 2}', '{"E": 5}'])
    generation = steering_eval.GenerationEvalScorer(
        scorer,
        judge,
        judge_model="gpt-4o-mini",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    prompt = steering_eval.TranscriptPrompt(
        prompt_id="p1",
        prompt="Say hello",
        trait_name="extraversion",
    )

    result = generation.evaluate_prompt(
        prompt,
        trait_name="extraversion",
        trait_code="E",
        alpha=1.0,
        max_new_tokens=8,
        temperature=0.1,
        top_p=0.9,
    )

    assert result.trait_expression_delta == 3.0
    assert len(judge.requests) == 2


def test_parse_alpha_grid_parses_comma_list():
    assert steering_eval.parse_alpha_grid("0.25, 0.5,1.0") == [0.25, 0.5, 1.0]


def test_measure_cross_trait_bleed_returns_source_target_matrix():
    prompts = {
        "extraversion": [
            steering_eval.PromptRecord(
                prompt_id="p1",
                question="Question",
                option_a="High",
                option_b="Low",
                option_a_is_high=True,
                option_b_is_high=False,
            )
        ]
    }
    scorer = FakeScorer(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ]
    )

    matrix = steering_eval.measure_cross_trait_bleed(
        [("extraversion", "E")],
        prompts,
        scorer,
        alpha=1.0,
    )

    assert matrix == {"extraversion": {"extraversion": 0.5}}

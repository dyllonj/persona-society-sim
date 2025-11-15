from pathlib import Path

from steering import eval as steering_eval


class FakeScorer(steering_eval.OptionScorer):
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.calls = []

    def score_options(self, prompt_text, option_texts, *, trait_code, alpha):
        self.calls.append((trait_code, alpha, prompt_text))
        return self.sequences.pop(0)

    def generate_text(self, prompt_text, *, trait_code, alpha, max_new_tokens, temperature, top_p):
        return f"{prompt_text}::{trait_code or 'none'}::{alpha}"


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
    assert len(result.prompt_results) == 2
    assert result.prompt_results[1].high_option == "B"


def test_canonicalize_trait_handles_aliases():
    assert steering_eval.canonicalize_trait("E") == ("extraversion", "E")
    assert steering_eval.canonicalize_trait("agreeableness") == ("agreeableness", "A")


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

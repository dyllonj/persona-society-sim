import math
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
    calls = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        self.calls.append(text)
        assert return_offsets_mapping
        if text == "prompt\nfirst":
            return {
                "input_ids": [1, 2, 3, 4],
                "offset_mapping": [(0, 6), (6, 7), (7, 9), (9, 12)],
            }
        if text == "prompt\nsecond":
            return {
                "input_ids": [1, 2, 5],
                "offset_mapping": [(0, 6), (6, 7), (7, 13)],
            }
        raise AssertionError(f"unexpected separately tokenized input: {text!r}")


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
    assert scorer.tokenizer.calls == ["prompt\nfirst", "prompt\nsecond"]
    assert scores[0].token_count == 2
    assert scores[1].token_count == 1
    assert scores[0].sum_logprob == pytest.approx(-2 * math.log(8))
    assert scores[1].sum_logprob == pytest.approx(-math.log(8))
    assert scores[0].mean_logprob == pytest.approx(scores[1].mean_logprob)


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
    assert result.alpha == 1.0


def test_evaluate_trait_dataset_uses_mean_logprob_as_primary_and_reports_sums():
    prompt = steering_eval.PromptRecord(
        prompt_id="p1",
        question="Question",
        option_a="Long high option",
        option_b="Short low option",
        option_a_is_high=True,
        option_b_is_high=False,
    )
    scorer = FakeScorer(
        [
            [
                steering_eval.ConditionalLogprob(-10.0, -1.0, 10),
                steering_eval.ConditionalLogprob(-2.0, -0.5, 4),
            ],
            [
                steering_eval.ConditionalLogprob(-6.0, -0.3, 20),
                steering_eval.ConditionalLogprob(-2.0, -0.5, 4),
            ],
        ]
    )

    result = steering_eval.evaluate_trait_dataset(
        "extraversion",
        "E",
        Path("heldout.jsonl"),
        [prompt],
        scorer,
        alpha=0.8,
        vector_store_id="E-v1",
        metadata_path=Path("E.meta.json"),
    )

    assert result.accuracy_baseline == 0.0
    assert result.accuracy_steered == 1.0
    assert result.logprob_gap_baseline == pytest.approx(-0.5)
    assert result.logprob_gap_steered == pytest.approx(0.2)
    assert result.summed_logprob_gap_baseline == pytest.approx(-8.0)
    assert result.summed_logprob_gap_steered == pytest.approx(-4.0)
    assert result.prompt_results[0].high_option_token_count == 10
    assert result.alpha == 0.8
    payload = steering_eval._trait_to_dict(result)
    assert payload["logprob_gap"]["metric"] == "mean_per_continuation_token"
    assert payload["summed_logprob_gap"]["metric"] == "sum_over_continuation_tokens"


def test_hf_scorer_uses_configured_dtype_without_downloading(monkeypatch):
    captured = {}

    class _Tokenizer:
        is_fast = True
        pad_token = "<pad>"
        eos_token = "<eos>"
        init_kwargs = {"_commit_hash": "b" * 40}

    class _Config:
        _commit_hash = "a" * 40
        hidden_size = 4
        num_hidden_layers = 2

    class _Model:
        config = _Config()

        def eval(self):
            return self

        @staticmethod
        def parameters():
            yield torch.zeros(1, dtype=torch.bfloat16)

    def fake_model_loader(*args, **kwargs):
        captured.update(kwargs)
        return _Model()

    monkeypatch.setattr(
        steering_eval.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(
        steering_eval.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_loader,
    )

    scorer = steering_eval.HFContrastScorer(
        "test/model",
        {},
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        dtype="bf16",
    )

    assert captured["torch_dtype"] is torch.bfloat16
    assert scorer.resolved_dtype == "bfloat16"


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


def test_parse_trait_alphas_accepts_codes_and_names_and_rejects_duplicates():
    assert steering_eval.parse_trait_alphas(["E=0.8,agreeableness=0.5", "C=0.6"]) == {
        "E": 0.8,
        "A": 0.5,
        "C": 0.6,
    }

    with pytest.raises(ValueError, match="Duplicate"):
        steering_eval.parse_trait_alphas(["E=0.8", "extraversion=1.0"])


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


def test_archival_provenance_hashes_runtime_and_every_input_class(tmp_path):
    prompt = tmp_path / "extraversion_eval.jsonl"
    prompt.write_text('{"id":"E1"}\n', encoding="utf-8")
    vector = tmp_path / "E-v1-layer1.npy"
    vector.write_bytes(b"vector")
    metadata = tmp_path / "E.meta.json"
    metadata.write_text('{"vector_store_id":"E-v1"}\n', encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text('{"vector_store_id":"E-v1"}\n', encoding="utf-8")
    vector_config = tmp_path / "steering.layers.yaml"
    vector_config.write_text("traits: {}\n", encoding="utf-8")

    class _Config:
        @staticmethod
        def to_dict():
            return {"hidden_size": 4, "num_hidden_layers": 2}

    class _Model:
        config = _Config()

    scorer = type(
        "Scorer",
        (),
        {
            "model": _Model(),
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "resolved_dtype": "bfloat16",
        },
    )()
    provenance = steering_eval.build_archival_provenance(
        scorer=scorer,
        prompt_paths={"extraversion": prompt},
        metadata_root=tmp_path,
        metadata_map={
            "E": {
                "resolved_vector_artifacts": {
                    "1": {
                        "path": str(vector),
                        "sha256": steering_eval._sha256_file(vector),
                    }
                }
            }
        },
        vector_config=vector_config,
    )

    assert provenance["runtime"]["dtype"] == "bfloat16"
    assert provenance["model"]["config_sha256"]
    assert provenance["files"]["prompts"]["extraversion"]["sha256"]
    assert provenance["files"]["vector_metadata"]["E"]["sha256"]
    assert provenance["files"]["vector_index"]["sha256"]
    assert provenance["files"]["vectors"]["E@1"]["sha256"]
    assert provenance["files"]["vector_config"]["sha256"]
    assert provenance["files"]["scripts"]["steering_eval"]["sha256"]


def test_report_content_hash_is_canonical_and_self_excluding():
    report = {
        "model": "test",
        "traits": [{"E": 1}],
        "alpha": 0.8,
        "generated_at": "2026-07-10T05:00:00+00:00",
    }
    first = steering_eval.report_content_sha256(report)
    report["report_content_sha256"] = first

    assert steering_eval.report_content_sha256(report) == first
    assert (
        steering_eval.report_content_sha256(
            {
                "generated_at": "2026-07-11T05:00:00+00:00",
                "alpha": 0.8,
                "traits": [{"E": 1}],
                "model": "test",
            }
        )
        == first
    )

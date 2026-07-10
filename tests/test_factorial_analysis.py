from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretability import analyze_factorial, build_factorial_prompts
from interpretability.common import sha256_file, sha256_json


def _eval_sources(tmp_path: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    scenarios = {
        "E": "Several unfamiliar attendees are waiting before a workshop begins.",
        "A": "A teammate made an error that delayed a shared task.",
        "C": "Several requests arrive with different deadlines.",
    }
    for stratum, filename in build_factorial_prompts.ORIGIN_FILES.items():
        path = tmp_path / filename
        path.write_text(
            json.dumps(
                {
                    "id": f"{stratum}101",
                    "question_text": scenarios[stratum],
                    "option_a": f"HIGH SECRET OPTION {stratum}",
                    "option_b": f"LOW SECRET OPTION {stratum}",
                    "option_a_is_high": True,
                    "option_b_is_high": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sources[stratum] = path
    return sources


def _prompt_bundle(tmp_path: Path) -> tuple[Path, list[dict]]:
    rows, provenance = build_factorial_prompts.build_prompt_records(
        _eval_sources(tmp_path), expected_per_stratum=1
    )
    path = tmp_path / "factorial-prompts.jsonl"
    build_factorial_prompts.write_prompt_bundle(path, rows, provenance)
    return path, rows


def test_prompt_builder_exposes_only_neutral_scenarios_and_stable_opaque_ids(
    tmp_path: Path,
) -> None:
    sources = _eval_sources(tmp_path)
    first, provenance = build_factorial_prompts.build_prompt_records(
        sources, expected_per_stratum=1
    )
    second, _ = build_factorial_prompts.build_prompt_records(sources, expected_per_stratum=1)

    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert [row["origin_stratum"] for row in first] == ["E", "A", "C"]
    assert all(row["prompt_id"].startswith("factorial-prompt-") for row in first)
    for row in first:
        assert "SECRET OPTION" not in row["text"]
        assert "option_a" not in row["text"]
        assert not build_factorial_prompts.FORBIDDEN_PROMPT_LABELS.search(row["text"])
        assert row["prompt_sha256"] == __import__("hashlib").sha256(
            row["text"].encode("utf-8")
        ).hexdigest()
    assert set(provenance["sources"]) == {"E", "A", "C"}

    output = tmp_path / "out" / "prompts.jsonl"
    manifest = build_factorial_prompts.write_prompt_bundle(output, first, provenance)
    assert manifest["prompts"] == 3
    assert manifest["output_sha256"] == sha256_file(output)
    assert not list(output.parent.glob(".*.tmp.*"))


def test_prompt_builder_enforces_counts_and_rejects_condition_cues(tmp_path: Path) -> None:
    sources = _eval_sources(tmp_path)
    with pytest.raises(ValueError, match="expected 20"):
        build_factorial_prompts.build_prompt_records(sources, expected_per_stratum=20)
    with pytest.raises(ValueError, match="forbidden condition cue"):
        build_factorial_prompts.render_structured_prompt(
            "Respond with high agreeableness when the teammate arrives."
        )


def _completion(action: str) -> str:
    return json.dumps(
        {"action": action, "params": {}, "utterance": f"I choose to {action}."},
        separators=(",", ":"),
    )


def _factorial_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    prompts_path, prompts = _prompt_bundle(tmp_path)
    completions = {
        "neutral": (_completion("talk"), [10, 20]),
        "E_only": (_completion("work"), [10, 21]),
        "A_only": ("not json", [99]),
        "C_only": (_completion("talk"), [10, 20]),
        "E_A_C": (_completion("scan"), [10, 20, 30]),
        "placebo_shuffled": ('{"action":"unknown","params":{},"utterance":"I wait."}', [77]),
    }
    events: list[dict] = []
    prompt_hashes: list[str] = []
    alphas = {"E": 0.0, "A": 0.0, "C": 0.0}
    for prompt_index, prompt in enumerate(prompts):
        prompt_hashes.append(prompt["prompt_sha256"])
        seed = 1000 + prompt_index
        for condition_index, condition in enumerate(analyze_factorial.CONDITION_ORDER):
            completion, generated_ids = completions[condition]
            events.append(
                {
                    "schema_version": "factorial-event-1.0",
                    "trace_id": f"trace-{prompt_index}-{condition}",
                    "run_id": "factorial-test",
                    "prompt_id": prompt["prompt_id"],
                    "prompt_index": prompt_index,
                    "origin_stratum": prompt["origin_stratum"],
                    "source_id": prompt["source_id"],
                    "condition": condition,
                    "condition_index": condition_index,
                    "prompt_hash": prompt["prompt_sha256"],
                    "prompt_text": prompt["text"],
                    "input_ids": [1, 2, prompt_index],
                    "attention_mask": [1, 1, 1],
                    "prompt_token_count": 3,
                    "generated_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "raw_completion": completion,
                    "paired_seed": seed,
                    "model_id": "tiny/model",
                    "model_revision": "a" * 40,
                    "tokenizer_revision": "a" * 40,
                    "inference_dtype": "fp32",
                    "quantization": None,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "effective_alphas": alphas,
                    "controller_alphas": {},
                    "vector_mode": "test",
                }
            )
    events_path = tmp_path / "factorial.jsonl"
    events_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in events
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "factorial-1.0",
        "run_id": "factorial-test",
        "model_id": "tiny/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "conditions": [
            {
                "name": condition,
                "effective_alphas": alphas,
                "controller_alphas": {},
                "vector_mode": "test",
            }
            for condition in analyze_factorial.CONDITION_ORDER
        ],
        "dtype": "fp32",
        "quantization": None,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "prompt_ids": [prompt["prompt_id"] for prompt in prompts],
        "prompt_hashes": prompt_hashes,
        "prompt_source_sha256": sha256_file(prompts_path),
        "prompts": len(prompts),
        "conditions_per_prompt": len(analyze_factorial.CONDITION_ORDER),
        "events": len(events),
        "output": events_path.name,
        "output_sha256": sha256_file(events_path),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    manifest_path = events_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return events_path, manifest_path, prompts_path, prompts


def test_factorial_analyzer_validates_pairs_and_reports_descriptive_contrasts(
    tmp_path: Path,
) -> None:
    events, manifest, prompts_path, prompts = _factorial_fixture(tmp_path)
    rubrics = tmp_path / "rubrics.jsonl"
    rubric_rows = []
    for prompt in prompts:
        for condition in analyze_factorial.CONDITION_ORDER:
            rubric_rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "condition": condition,
                    "rubric_id": "human-rubric-v1",
                    "score": 2.0 if condition == "E_only" else 1.0,
                }
            )
    rubrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rubric_rows), encoding="utf-8"
    )

    report = analyze_factorial.build_analysis(
        events_path=events,
        manifest_path=manifest,
        prompt_metadata_path=prompts_path,
        rubric_scores_path=rubrics,
    )

    assert report["prompt_seed_pairs"] == 3
    assert report["generations"] == 18
    assert report["scope"]["personality_scores_constructed"] is False
    assert report["scope"]["model_judge_calls"] is False
    assert report["arms"]["A_only"]["json_syntax_valid_rate"] == 0.0
    assert report["arms"]["placebo_shuffled"]["json_syntax_valid_rate"] == 1.0
    assert report["arms"]["placebo_shuffled"]["structured_action_valid_rate"] == 0.0
    assert report["arms"]["C_only"]["exact_path_divergence_neutral_rate"] == 0.0
    assert report["paired_contrasts"]["E_only"]["exact_path_divergence_rate"] == 1.0
    assert report["paired_contrasts"]["E_only"]["mean_external_rubric_score_delta"] == 1.0
    assert set(report["origin_strata"]) == {"E", "A", "C"}
    assert all(value["prompt_seed_pairs"] == 1 for value in report["origin_strata"].values())
    assert len(report["per_prompt_seed_contrasts"]) == 15

    output_prefix = tmp_path / "analysis" / "factorial"
    json_path, markdown_path = analyze_factorial._write_outputs(output_prefix, report)
    assert json.loads(json_path.read_text(encoding="utf-8"))["generations"] == 18
    assert "descriptive paired-prompt diagnostic" in markdown_path.read_text(encoding="utf-8")
    assert not list(output_prefix.parent.glob(".*.tmp.*"))


def test_factorial_analyzer_rejects_tampered_events_and_manifest(tmp_path: Path) -> None:
    events, manifest, prompts_path, _ = _factorial_fixture(tmp_path)
    with events.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="factorial output hash mismatch"):
        analyze_factorial.build_analysis(
            events_path=events,
            manifest_path=manifest,
            prompt_metadata_path=prompts_path,
        )


def test_structured_action_parser_distinguishes_json_from_contract_validity() -> None:
    unsupported = analyze_factorial.parse_structured_action(
        '{"action":"dance","params":{},"utterance":"I dance."}'
    )
    malformed = analyze_factorial.parse_structured_action("not json")
    valid = analyze_factorial.parse_structured_action(_completion("scan"))

    assert unsupported["json_syntax_valid"] is True
    assert unsupported["structured_action_valid"] is False
    assert malformed["json_syntax_valid"] is False
    assert valid["structured_action_valid"] is True

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from interpretability.common import sha256_json
from interpretability.run_factorial import (
    CONDITION_ORDER,
    TRAITS,
    FactorialPrompt,
    VectorBundle,
    _active_vector_fields,
    _canonical_rows_sha256,
    _progress_payload,
    _write_jsonl_atomic,
    build_final_manifest,
    build_conditions,
    condition_vector_provenance,
    enforce_output_policy,
    factorial_artifact_paths,
    load_factorial_prompts,
    load_resume_rows,
    load_vector_bundle,
    parse_base_alphas,
    persist_prompt_block,
    placebo_permutation_seed,
    prompt_seed,
    shuffled_vector,
    validate_resume_rows,
    validate_neutral_prompts,
)


def test_factorial_conditions_are_complete_and_do_not_leak_prior_arm_alphas():
    base = parse_base_alphas("E=0.8,A=0.5,C=0.6")
    conditions = build_conditions(base)

    assert tuple(condition.name for condition in conditions) == CONDITION_ORDER
    assert conditions[0].effective_alphas == {"E": 0.0, "A": 0.0, "C": 0.0}
    controller_keys = {
        controller_trait
        for trait in TRAITS
        for controller_trait in (trait, f"placebo_{trait}")
    }
    assert all(set(condition.controller_alphas) == controller_keys for condition in conditions)

    for condition in conditions[:-1]:
        assert all(
            condition.controller_alphas[f"placebo_{trait}"] == 0.0 for trait in TRAITS
        )
    placebo = conditions[-1]
    assert all(placebo.controller_alphas[trait] == 0.0 for trait in TRAITS)
    assert {placebo.controller_alphas[f"placebo_{trait}"] for trait in TRAITS} == {
        0.8,
        0.5,
        0.6,
    }


def test_factorial_alpha_parser_requires_all_three_finite_nonzero_traits():
    assert parse_base_alphas("C=-0.2,E=1,A=0.5") == {"E": 1.0, "A": 0.5, "C": -0.2}

    with pytest.raises(ValueError, match="missing base alphas"):
        parse_base_alphas("E=1,A=1")
    with pytest.raises(ValueError, match="finite and nonzero"):
        parse_base_alphas("E=1,A=1,C=0")
    with pytest.raises(ValueError, match="duplicate"):
        parse_base_alphas("E=1,A=1,C=1,E=2")


def test_prompt_and_placebo_seeds_are_stable_and_pairable():
    assert prompt_seed(17, 2) == prompt_seed(17, 2)
    assert prompt_seed(17, 2) != prompt_seed(17, 3)

    source_hash = "a" * 64
    seed = placebo_permutation_seed(29, "E", 36, source_hash)
    assert seed == placebo_permutation_seed(29, "E", 36, source_hash)
    assert seed != placebo_permutation_seed(29, "A", 36, source_hash)


def test_coordinate_shuffled_placebo_is_deterministic_and_norm_preserving():
    vector = torch.arange(16, dtype=torch.float32)
    first = shuffled_vector(vector, seed=991)
    second = shuffled_vector(vector, seed=991)

    assert torch.equal(first, second)
    assert not torch.equal(first, vector)
    assert torch.equal(first.sort().values, vector.sort().values)
    assert torch.linalg.vector_norm(first) == torch.linalg.vector_norm(vector)


def test_factorial_prompts_reject_explicit_persona_labels():
    validate_neutral_prompts(
        [
            "A team finds conflicting measurements. Return the next evidence-checking action.",
            "A deadline moved forward by two days. Decide what to do next.",
        ]
    )

    with pytest.raises(ValueError, match="forbidden persona label"):
        validate_neutral_prompts(["Act with high agreeableness and answer the request."])
    with pytest.raises(ValueError, match="forbidden persona label"):
        validate_neutral_prompts(["Use this personality trait when deciding."])


def test_factorial_prompt_loader_preserves_ids_and_hidden_strata(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    text = "Choose one action in response to a scheduling conflict."
    prompt_path.write_text(
        json.dumps(
            {
                "prompt_id": "stable-prompt",
                "text": text,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "origin_stratum": "C",
                "source_id": "C101",
                "source_file": "conscientiousness_eval.jsonl",
                "source_record_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_factorial_prompts(prompt_path)

    assert prompts[0].prompt_id == "stable-prompt"
    assert prompts[0].origin_stratum == "C"
    assert prompts[0].source_id == "C101"


def test_vector_bundle_enforces_model_and_polarity_and_builds_hashed_placebos(tmp_path):
    config_dir = tmp_path / "configs"
    vector_dir = tmp_path / "vectors"
    config_dir.mkdir()
    vector_dir.mkdir()
    index_rows = []
    traits_config = {}
    raw_vectors = {
        "E": np.array([1, 2, 3, 4], dtype=np.float32),
        "A": np.array([2, 3, 5, 7], dtype=np.float32),
        "C": np.array([11, 13, 17, 19], dtype=np.float32),
    }
    for layer, trait in enumerate(TRAITS):
        vector_id = f"{trait}_test_v1"
        filename = f"{vector_id}_layer{layer}.npy"
        np.save(vector_dir / filename, raw_vectors[trait])
        index_rows.append(
            {
                "vector_store_id": vector_id,
                "trait": trait,
                "method": "caa_ab",
                "layer_id": layer,
                "vector_path": filename,
                "train_set_hash": trait.lower() * 64,
            }
        )
        traits_config[trait] = {
            "vector_store_id": vector_id,
            "layers": [layer],
            "polarity": -1.0 if trait == "E" else 1.0,
        }
    (vector_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    metadata_path = config_dir / "steering.layers.yaml"
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "vector_root": "../vectors",
                "defaults": {"model": "test/model", "num_hidden_layers": 3},
                "traits": traits_config,
            }
        ),
        encoding="utf-8",
    )

    bundle = load_vector_bundle(
        metadata_path,
        model_id="test/model",
        expected_width=4,
        expected_layers=3,
        placebo_seed=313,
    )

    assert torch.equal(bundle.vectors["E"][0], -torch.from_numpy(raw_vectors["E"]))
    assert torch.equal(
        bundle.vectors["placebo_E"][0].sort().values,
        bundle.vectors["E"][0].sort().values,
    )
    assert bundle.applied_hashes["E"]["0"] != bundle.applied_hashes["placebo_E"]["0"]
    assert bundle.mapping["placebo_E"]["layers"]["0"]["permutation_seed"]

    placebo = build_conditions({"E": 1, "A": 1, "C": 1})[-1]
    provenance = condition_vector_provenance(placebo, bundle)
    assert set(provenance) == {"placebo_E", "placebo_A", "placebo_C"}

    with pytest.raises(ValueError, match="metadata model mismatch"):
        load_vector_bundle(
            metadata_path,
            model_id="wrong/model",
            expected_width=4,
            expected_layers=3,
            placebo_seed=313,
        )


def _durability_fixture(prompt_count: int = 2):
    conditions = build_conditions({"E": 0.8, "A": 0.5, "C": 0.6})
    prompts = [
        FactorialPrompt(
            prompt_id=f"stable-{index}",
            text=f"Prompt text {index}",
            origin_stratum="E" if index % 2 == 0 else "A",
            source_id=f"source-{index}",
            source_file="eval.jsonl",
            source_record_index=index,
        )
        for index in range(prompt_count)
    ]
    prompt_tokens = [([1, index + 2], [1, 1]) for index in range(prompt_count)]
    vector_ids = {trait: f"{trait}-vectors" for trait in TRAITS}
    source_hashes = {trait: {str(index): trait.lower() * 64} for index, trait in enumerate(TRAITS)}
    applied_hashes = {
        **{trait: {str(index): (trait.lower() + "i") * 32} for index, trait in enumerate(TRAITS)},
        **{
            f"placebo_{trait}": {str(index): (trait.lower() + "p") * 32}
            for index, trait in enumerate(TRAITS)
        },
    }
    mapping = {
        **{trait: {"mode": "identity", "source_trait": trait} for trait in TRAITS},
        **{
            f"placebo_{trait}": {"mode": "coordinate_permutation", "source_trait": trait}
            for trait in TRAITS
        },
    }
    bundle = VectorBundle(
        vectors={},
        vector_ids=vector_ids,
        source_hashes=source_hashes,
        applied_hashes=applied_hashes,
        polarities={trait: 1.0 for trait in TRAITS},
        mapping=mapping,
        metadata_sha256="m" * 64,
        index_sha256="n" * 64,
    )
    run_spec = {
        "schema_version": "factorial-1.0",
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "model_config_sha256": "c" * 64,
        "dtype": "float32",
        "quantization": None,
        "conditions": [
            {
                "name": condition.name,
                "effective_alphas": condition.effective_alphas,
                "controller_alphas": condition.controller_alphas,
                "vector_mode": condition.vector_mode,
            }
            for condition in conditions
        ],
        "base_seed": 17,
        "placebo_seed": 29,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 10,
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "prompt_hashes": [hashlib.sha256(prompt.text.encode()).hexdigest() for prompt in prompts],
        "vector_metadata_sha256": bundle.metadata_sha256,
        "vector_index_sha256": bundle.index_sha256,
        "vector_ids": bundle.vector_ids,
        "vector_source_hashes": bundle.source_hashes,
        "vector_applied_hashes": bundle.applied_hashes,
        "vector_polarities": bundle.polarities,
        "code_hashes": {
            "run_factorial.py": "r" * 64,
            "steering/hooks.py": "h" * 64,
        },
    }
    run_id = f"factorial-{sha256_json(run_spec)[:16]}"

    def decode(ids):
        return ",".join(str(token) for token in ids)

    rows = []
    for prompt_index, prompt in enumerate(prompts):
        seed = prompt_seed(run_spec["base_seed"], prompt_index)
        prompt_hash = hashlib.sha256(prompt.text.encode()).hexdigest()
        for condition_index, condition in enumerate(conditions):
            generated_ids = [100 + prompt_index, condition_index]
            vector_ids_for_arm, source_for_arm, applied_for_arm = _active_vector_fields(
                condition, bundle
            )
            trace_id = (
                f"{run_id}-p{prompt_index:04d}-{condition.name}-"
                f"{sha256_json([prompt_hash, condition.name, seed, generated_ids])[:12]}"
            )
            rows.append(
                {
                    "schema_version": "factorial-event-1.0",
                    "trace_id": trace_id,
                    "run_id": run_id,
                    "prompt_id": prompt.prompt_id,
                    "prompt_index": prompt_index,
                    "origin_stratum": prompt.origin_stratum,
                    "source_id": prompt.source_id,
                    "source_file": prompt.source_file,
                    "source_record_index": prompt.source_record_index,
                    "condition": condition.name,
                    "condition_index": condition_index,
                    "prompt_hash": prompt_hash,
                    "prompt_text": prompt.text,
                    "input_ids": prompt_tokens[prompt_index][0],
                    "attention_mask": prompt_tokens[prompt_index][1],
                    "prompt_token_count": len(prompt_tokens[prompt_index][0]),
                    "generated_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "raw_completion": decode(generated_ids),
                    "model_id": run_spec["model_id"],
                    "model_revision": run_spec["model_revision"],
                    "tokenizer_revision": run_spec["tokenizer_revision"],
                    "inference_dtype": run_spec["dtype"],
                    "quantization": None,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "sampling_seed": seed,
                    "paired_seed": seed,
                    "effective_alphas": condition.effective_alphas,
                    "controller_alphas": condition.controller_alphas,
                    "steering_applied": any(
                        alpha != 0.0 for alpha in condition.controller_alphas.values()
                    ),
                    "vector_mode": condition.vector_mode,
                    "steering_vector_ids": vector_ids_for_arm,
                    "steering_vector_hashes": source_for_arm,
                    "applied_vector_hashes": applied_for_arm,
                    "vector_mapping": condition_vector_provenance(condition, bundle),
                }
            )
    return run_spec, run_id, prompts, prompt_tokens, conditions, bundle, decode, rows


def test_factorial_output_policy_requires_explicit_resume(tmp_path):
    paths = factorial_artifact_paths(tmp_path / "factorial.jsonl")
    paths.output.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --resume"):
        enforce_output_policy(paths, resume=False)
    with pytest.raises(FileNotFoundError, match="progress sidecar"):
        enforce_output_policy(paths, resume=True)


def test_resume_accepts_only_complete_unique_internally_valid_blocks():
    run_spec, run_id, prompts, prompt_tokens, conditions, bundle, decode, rows = (
        _durability_fixture()
    )
    kwargs = {
        "run_spec": run_spec,
        "run_id": run_id,
        "prompts": prompts,
        "prompt_tokens": prompt_tokens,
        "conditions": conditions,
        "bundle": bundle,
        "decode_generated": decode,
    }

    assert validate_resume_rows(rows, **kwargs) == 2
    with pytest.raises(ValueError, match="partial prompt block"):
        validate_resume_rows(rows[:-1], **kwargs)

    corrupt = [dict(row) for row in rows]
    corrupt[1]["condition"] = "neutral"
    with pytest.raises(ValueError, match="disagrees on condition"):
        validate_resume_rows(corrupt, **kwargs)

    corrupt = [dict(row) for row in rows]
    corrupt[6]["generated_ids"] = list(corrupt[0]["generated_ids"])
    corrupt[6]["raw_completion"] = decode(corrupt[6]["generated_ids"])
    with pytest.raises(ValueError, match="non-reproducible trace_id"):
        validate_resume_rows(corrupt, **kwargs)


def test_resume_reconciles_exactly_one_output_first_progress_lag(tmp_path):
    run_spec, run_id, prompts, prompt_tokens, conditions, bundle, decode, rows = (
        _durability_fixture()
    )
    paths = factorial_artifact_paths(tmp_path / "factorial.jsonl")
    started_at = "2026-01-01T00:00:00+00:00"
    progress = _progress_payload(
        run_spec=run_spec,
        run_id=run_id,
        rows=[],
        started_at=started_at,
        status="active",
    )
    paths.progress.write_text(json.dumps(progress), encoding="utf-8")
    _write_jsonl_atomic(paths.output, rows[: len(CONDITION_ORDER)])

    recovered, recovered_started_at = load_resume_rows(
        paths,
        run_spec=run_spec,
        run_id=run_id,
        prompts=prompts,
        prompt_tokens=prompt_tokens,
        conditions=conditions,
        bundle=bundle,
        decode_generated=decode,
    )

    assert recovered == rows[: len(CONDITION_ORDER)]
    assert recovered_started_at == started_at
    reconciled = json.loads(paths.progress.read_text(encoding="utf-8"))
    assert reconciled["completed_prompt_blocks"] == 1
    assert reconciled["output_sha256"] == _canonical_rows_sha256(recovered)

    incompatible = dict(run_spec)
    incompatible["temperature"] = 0.8
    with pytest.raises(ValueError, match="run specification is incompatible"):
        load_resume_rows(
            paths,
            run_spec=incompatible,
            run_id=run_id,
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            conditions=conditions,
            bundle=bundle,
            decode_generated=decode,
        )


def test_resume_rejects_progress_more_than_one_block_behind(tmp_path):
    run_spec, run_id, prompts, prompt_tokens, conditions, bundle, decode, rows = (
        _durability_fixture()
    )
    paths = factorial_artifact_paths(tmp_path / "factorial.jsonl")
    paths.progress.write_text(
        json.dumps(
            _progress_payload(
                run_spec=run_spec,
                run_id=run_id,
                rows=[],
                started_at="2026-01-01T00:00:00+00:00",
                status="active",
            )
        ),
        encoding="utf-8",
    )
    _write_jsonl_atomic(paths.output, rows)

    with pytest.raises(ValueError, match="lags durable output by more than one"):
        load_resume_rows(
            paths,
            run_spec=run_spec,
            run_id=run_id,
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            conditions=conditions,
            bundle=bundle,
            decode_generated=decode,
        )


def test_persist_rejects_partial_block_before_touching_output(tmp_path):
    run_spec, run_id, *_rest, rows = _durability_fixture(prompt_count=1)
    paths = factorial_artifact_paths(tmp_path / "factorial.jsonl")

    with pytest.raises(ValueError, match="complete six-arm"):
        persist_prompt_block(
            paths,
            rows=rows[:-1],
            run_spec=run_spec,
            run_id=run_id,
            started_at="2026-01-01T00:00:00+00:00",
        )

    assert not paths.output.exists()
    assert not paths.progress.exists()


def test_final_manifest_is_timestamp_free_and_deterministic(tmp_path):
    run_spec, run_id, *_rest, rows = _durability_fixture(prompt_count=1)
    output = tmp_path / "factorial.jsonl"
    _write_jsonl_atomic(output, rows)

    first = build_final_manifest(
        run_spec=run_spec,
        run_id=run_id,
        output=output,
        prompts=1,
        conditions_per_prompt=len(CONDITION_ORDER),
    )
    second = build_final_manifest(
        run_spec=run_spec,
        run_id=run_id,
        output=output,
        prompts=1,
        conditions_per_prompt=len(CONDITION_ORDER),
    )

    assert first == second
    assert "created_at" not in first
    assert first["output_sha256"] == _canonical_rows_sha256(rows)

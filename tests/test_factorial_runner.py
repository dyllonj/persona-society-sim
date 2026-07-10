from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from interpretability.run_factorial import (
    CONDITION_ORDER,
    TRAITS,
    build_conditions,
    condition_vector_provenance,
    load_factorial_prompts,
    load_vector_bundle,
    parse_base_alphas,
    placebo_permutation_seed,
    prompt_seed,
    shuffled_vector,
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

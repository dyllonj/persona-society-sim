from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.analyze_society_study import (
    aggregate_one_run,
    paired_seed_contrasts,
    summarize_arm_outcomes,
)
from scripts.run_society_study import (
    PROJECT_ROOT,
    build_run_specs,
    enforce_real_run_budget,
    estimate_plan,
    execute_specs,
    load_matrix,
    materialize_spec,
    sha256_file,
)
from orchestrator.cli import shuffle_trait_vectors


MATRIX_PATH = PROJECT_ROOT / "experiments/society_study/matrix.yaml"


def test_runtime_placebo_is_a_derangement_without_fixed_traits():
    vectors = {
        trait: {0: np.array([index], dtype=np.float32)}
        for index, trait in enumerate(("E", "A", "C"), 1)
    }
    norms = {trait: {0: float(index)} for index, trait in enumerate(vectors, 1)}

    shuffled, shuffled_norms, mapping = shuffle_trait_vectors(
        vectors, norms, random.Random(101)
    )

    assert all(target != source for target, source in mapping.items())
    assert set(mapping.values()) == set(vectors)
    for target, source in mapping.items():
        assert shuffled[target][0] is vectors[source][0]
        assert shuffled_norms[target] == norms[source]


def _write_rows(root: Path, kind: str, rows: list[dict]) -> None:
    directory = root / kind
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / f"{kind}_t00001.parquet")


def test_society_matrix_builds_thirty_preregistered_run_configs(tmp_path):
    matrix = load_matrix(MATRIX_PATH)
    specs = build_run_specs(
        matrix,
        matrix_path=MATRIX_PATH,
        output_root=tmp_path,
    )

    assert len(specs) == 30
    assert {(spec.arm, spec.seed) for spec in specs} == {
        (arm, seed) for arm in matrix["arms"] for seed in matrix["seeds"]
    }
    for spec in specs:
        assert spec.config["population"] == 30
        assert spec.config["steps"] == 100
        assert spec.config["inference"]["persona_prompt"] is False
        assert spec.config["inference"]["structured_actions"] is True
        assert spec.config["model_revision"] == matrix["model"]["revision"]
    placebo = next(spec for spec in specs if spec.arm == "placebo" and spec.seed == 303)
    mapping = placebo.config["study"]["placebo_vector_mapping"]
    assert set(mapping) == {"E", "A", "C"}
    assert set(mapping.values()) == {"E", "A", "C"}
    assert all(target != source for target, source in mapping.items())
    assert placebo.config["study"]["placebo_algorithm"] == (
        "trait_label_derangement_v1"
    )

    plan = estimate_plan(specs, matrix, mock_model=False)
    assert plan["n_runs"] == 30
    assert plan["estimated_generations_upper_bound"] == 90_000
    assert plan["estimated_tokens"] == 85_320_000
    assert plan["estimated_gpu_hours"] == pytest.approx(50.0)


def test_real_execution_budget_gate_rejects_missing_or_insufficient_cap():
    plan = {"estimated_gpu_hours": 50.0}
    with pytest.raises(ValueError, match="requires --max-estimated-gpu-hours"):
        enforce_real_run_budget(
            plan,
            mock_model=False,
            max_estimated_gpu_hours=None,
            hourly_rate_usd=None,
            max_estimated_cost_usd=None,
        )
    with pytest.raises(ValueError, match="exceeds the authorized cap"):
        enforce_real_run_budget(
            plan,
            mock_model=False,
            max_estimated_gpu_hours=10.0,
            hourly_rate_usd=None,
            max_estimated_cost_usd=None,
        )
    with pytest.raises(ValueError, match="estimated \\$100.00 cost"):
        enforce_real_run_budget(
            plan,
            mock_model=False,
            max_estimated_gpu_hours=50.0,
            hourly_rate_usd=2.0,
            max_estimated_cost_usd=20.0,
        )

    enforce_real_run_budget(
        plan,
        mock_model=False,
        max_estimated_gpu_hours=50.0,
        hourly_rate_usd=2.0,
        max_estimated_cost_usd=100.0,
    )
    assert plan["estimated_cost_usd"] == 100.0


def test_materialized_run_config_and_manifest_are_immutable(tmp_path):
    matrix = load_matrix(MATRIX_PATH)
    spec = build_run_specs(
        matrix,
        matrix_path=MATRIX_PATH,
        output_root=tmp_path,
        only_arms={"e_only"},
        only_seeds={101},
    )[0]
    base_config = PROJECT_ROOT / matrix["base_config"]

    manifest = materialize_spec(
        spec,
        matrix_path=MATRIX_PATH,
        base_config_path=base_config,
    )
    assert materialize_spec(
        spec,
        matrix_path=MATRIX_PATH,
        base_config_path=base_config,
    ) == manifest
    config_path = spec.run_dir / "config.yaml"
    assert sha256_file(config_path) == manifest["config_sha256"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["study"]["arm"] == "e_only"
    assert config["steering"]["active_traits"] == ["E"]

    config_path.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable study artifact differs"):
        materialize_spec(
            spec,
            matrix_path=MATRIX_PATH,
            base_config_path=base_config,
        )


def test_execute_gate_path_uses_mock_model_and_resume_skips_completion(tmp_path):
    matrix = load_matrix(MATRIX_PATH)
    spec = build_run_specs(
        matrix,
        matrix_path=MATRIX_PATH,
        output_root=tmp_path,
        only_arms={"neutral"},
        only_seeds={101},
    )[0]
    manifest = materialize_spec(
        spec,
        matrix_path=MATRIX_PATH,
        base_config_path=PROJECT_ROOT / matrix["base_config"],
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], _cwd: Path, log_path: Path) -> int:
        commands.append(command)
        log_path.write_text("mock run\n", encoding="utf-8")
        _write_rows(
            spec.run_dir / "outputs",
            "actions",
            [{"run_id": spec.run_id, "tick": 0, "outcome": "success"}],
        )
        return 0

    result = execute_specs(
        [spec],
        {spec.run_id: manifest},
        mock_model=True,
        resume=False,
        run_command=fake_run,
    )
    assert result["completed"] == [spec.run_id]
    assert "--mock-model" in commands[0]
    assert execute_specs(
        [spec],
        {spec.run_id: manifest},
        mock_model=True,
        resume=True,
        run_command=fake_run,
    )["skipped"] == [spec.run_id]
    assert len(commands) == 1


def test_aggregate_one_run_collapses_ticks_and_actions_before_inference(tmp_path):
    run_id = "run-a"
    _write_rows(
        tmp_path,
        "metrics_snapshots",
        [
            {
                "run_id": run_id,
                "tick": 0,
                "trait_key": "global",
                "cooperation_rate": 0.2,
                "gini_wealth": 0.1,
                "polarization_modularity": 0.3,
                "conflicts": 1,
                "rule_enforcement_cost": 2.0,
            },
            {
                "run_id": run_id,
                "tick": 1,
                "trait_key": "global",
                "cooperation_rate": 0.6,
                "gini_wealth": 0.3,
                "polarization_modularity": 0.5,
                "conflicts": 2,
                "rule_enforcement_cost": 3.0,
            },
        ],
    )
    _write_rows(
        tmp_path,
        "actions",
        [
            {
                "run_id": run_id,
                "tick": 0,
                "action_type": "research",
                "outcome": "success",
            },
            {
                "run_id": run_id,
                "tick": 1,
                "action_type": "submit_report",
                "outcome": "fail",
            },
        ],
    )
    _write_rows(
        tmp_path,
        "messages",
        [{"run_id": run_id, "tick": 0, "tokens_in": 10, "tokens_out": 5}],
    )

    outcome = aggregate_one_run(
        run_id=run_id,
        arm="e_only",
        seed=101,
        output_dir=tmp_path,
    )

    assert outcome["analysis_unit"] == "simulation_run"
    assert outcome["cooperation_rate"] == pytest.approx(0.4)
    assert outcome["conflicts_total"] == 3
    assert outcome["action_success_rate"] == 0.5
    assert outcome["tokens_total"] == 15


def test_arm_intervals_and_contrasts_use_equal_weight_run_rows():
    per_run = [
        {
            "analysis_unit": "simulation_run",
            "arm": "neutral",
            "seed": 1,
            "cooperation_rate": 0.0,
            "n_actions": 1_000_000,
        },
        {
            "analysis_unit": "simulation_run",
            "arm": "neutral",
            "seed": 2,
            "cooperation_rate": 0.2,
            "n_actions": 1,
        },
        {
            "analysis_unit": "simulation_run",
            "arm": "e_only",
            "seed": 1,
            "cooperation_rate": 0.4,
            "n_actions": 1,
        },
        {
            "analysis_unit": "simulation_run",
            "arm": "e_only",
            "seed": 2,
            "cooperation_rate": 0.8,
            "n_actions": 1_000_000,
        },
    ]

    summary = summarize_arm_outcomes(per_run, ["cooperation_rate"])
    neutral = next(row for row in summary if row["arm"] == "neutral")
    e_only = next(row for row in summary if row["arm"] == "e_only")
    assert neutral["mean"] == pytest.approx(0.1)
    assert e_only["mean"] == pytest.approx(0.6)
    assert neutral["n_runs"] == 2
    assert neutral["ci95_low"] is not None

    contrasts = paired_seed_contrasts(
        per_run,
        ["cooperation_rate"],
        [["e_only", "neutral"]],
    )
    assert contrasts[0]["analysis_unit"] == "paired_simulation_seed"
    assert contrasts[0]["paired_seeds"] == [1, 2]
    assert contrasts[0]["mean"] == pytest.approx(0.5)

import json

import pytest

from steering.steerability_smoke_test import (
    _cli,
    directional_agreement,
    gate_activation_diffs,
    gate_preliminary_steerability,
    mean_cosine_directional_agreement,
    project_activation_diff,
)


def test_directional_agreement_counts_vector_projections():
    summary = directional_agreement(
        [[1.0, 0.0], [0.5, 0.2], [-0.1, 0.0], [0.0, 0.0]],
        reference_vector=[1.0, 0.0],
        deadband=0.0,
    )

    assert summary.count == 4
    assert summary.aligned == 2
    assert summary.opposed == 1
    assert summary.neutral == 1
    assert summary.agreement == 0.5


def test_project_activation_diff_requires_reference_for_vectors():
    with pytest.raises(ValueError, match="reference_vector"):
        project_activation_diff([1.0, 2.0])


def test_gate_activation_diffs_returns_pass_warn_fail():
    passed = gate_activation_diffs(
        [0.2, 0.3, 0.4, -0.1],
        pass_threshold=0.75,
        warn_threshold=0.5,
    )
    warned = gate_activation_diffs(
        [0.2, 0.3, -0.4, -0.1],
        pass_threshold=0.75,
        warn_threshold=0.5,
    )
    failed = gate_activation_diffs(
        [0.2, -0.3, -0.4, -0.1],
        pass_threshold=0.75,
        warn_threshold=0.5,
    )

    assert passed.status == "PASS"
    assert warned.status == "WARN"
    assert failed.status == "FAIL"
    assert any("warn_threshold" in reason for reason in failed.reasons)


def test_preliminary_steerability_uses_mean_cosine_agreement():
    score = mean_cosine_directional_agreement(
        [[1.0, 0.0], [0.9, 0.1], [1.1, -0.1], [0.8, 0.2], [1.2, -0.2]]
    )
    gate = gate_preliminary_steerability(
        trait="E",
        layer=12,
        activation_diffs=[
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
            [0.8, 0.2],
            [1.2, -0.2],
        ],
        pass_threshold=0.3,
        warn_threshold=0.2,
        min_count=5,
    )

    assert score > 0.9
    assert gate.status == "PASS"
    assert gate.preliminary_steerability == pytest.approx(score)


def test_cli_gates_json_payload_without_hf_model(tmp_path, capsys):
    diff_path = tmp_path / "diffs.json"
    diff_path.write_text(
        json.dumps({"activation_diffs": [0.4, 0.2, -0.1]}),
        encoding="utf-8",
    )

    exit_code = _cli(
        [
            "--diffs",
            str(diff_path),
            "--pass-threshold",
            "0.6",
            "--warn-threshold",
            "0.4",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "PASS"
    assert output["summary"]["agreement"] == pytest.approx(2 / 3)

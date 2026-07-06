from pathlib import Path

from steering.ci_checks import (
    TraitCurvePoint,
    TraitDirectionality,
    cosine_similarity,
    validate_cosine_stability,
    validate_anti_steerable_fraction,
    validate_bleed_matrix,
    validate_directional_agreement_metadata,
    validate_directionality,
    validate_monotonic_logprobs,
)


def test_cosine_similarity_rejects_misaligned_vectors():
    vectors = {
        "E": {1: [(0, [1.0, 0.0]), (1, [0.0, 1.0])]}
    }
    failures = validate_cosine_stability(vectors, threshold=0.95)
    assert failures
    assert "cosine=0.0000" in failures[0]


def test_cosine_similarity_handles_zero_norm():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_directionality_gates_sign_and_improvement():
    points = [
        TraitDirectionality(
            trait="agreeableness",
            seed=0,
            alpha=1.0,
            sign_consistency=0.4,
            directional_improvement=0.3,
            source=Path("report.json"),
        )
    ]
    failures = validate_directionality(points, sign_threshold=0.5, directional_threshold=0.6)
    assert len(failures) == 2
    assert "sign_consistency" in failures[0]
    assert "directional_improvement" in failures[1]


def test_monotonic_logprob_validation_detects_regression():
    curve = [
        TraitCurvePoint(
            trait="extraversion",
            seed=42,
            alpha=0.5,
            logprob_gap_delta=0.2,
            source=Path("report.json"),
        ),
        TraitCurvePoint(
            trait="extraversion",
            seed=42,
            alpha=1.0,
            logprob_gap_delta=0.1,
            source=Path("report.json"),
        ),
    ]
    failures = validate_monotonic_logprobs(curve, tolerance=1e-6)
    assert failures
    assert "0.2000->0.1000" in failures[0]


def test_validate_anti_steerable_fraction_flags_high_values():
    failures = validate_anti_steerable_fraction(
        [{"trait_name": "extraversion", "anti_steerable_fraction": 0.75}],
        threshold=0.5,
    )

    assert failures
    assert "extraversion" in failures[0]


def test_validate_directional_agreement_metadata_flags_weak_layers():
    failures = validate_directional_agreement_metadata(
        {
            "trait": "E",
            "layers": [
                {"layer_id": 12, "directional_agreement": 0.2},
                {"layer_id": 36, "directional_agreement": 0.8},
            ],
        },
        threshold=0.3,
    )

    assert failures == ["E layer=12 directional_agreement=0.200 < 0.300"]


def test_validate_bleed_matrix_flags_off_diagonal_bleed():
    failures = validate_bleed_matrix(
        {"E": {"E": 3.0, "A": 2.5}, "A": {"E": 0.5, "A": 1.0}},
        threshold=2.0,
    )

    assert failures == ["bleed E->A=2.500 > 2.000"]

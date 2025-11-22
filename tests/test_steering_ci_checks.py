from pathlib import Path

from steering.ci_checks import (
    TraitCurvePoint,
    TraitDirectionality,
    cosine_similarity,
    validate_cosine_stability,
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

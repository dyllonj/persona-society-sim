from agents.language_backend import MockBackend
from steering.per_trait_strength import recommend_pareto_optimal_alphas, resolve_per_trait_strength


def test_resolve_per_trait_strength_fallback_only():
    scaled = resolve_per_trait_strength(
        {"E": 0.8, "A": -0.4},
        per_trait_strength={"fallback": 0.5},
    )

    assert scaled == {"E": 0.4, "A": -0.2}


def test_resolve_per_trait_strength_per_trait_only():
    scaled = resolve_per_trait_strength(
        {"E": 0.8, "A": -0.4, "C": 0.25},
        per_trait_strength={"E": 0.25, "A": 2.0},
    )

    assert scaled == {"E": 0.2, "A": -0.8, "C": 0.25}


def test_language_backend_applies_global_and_per_trait_strength():
    backend = MockBackend(
        alpha_strength=2.0,
        per_trait_strength={"fallback": 0.5, "E": 0.25},
    )

    scaled = backend._scale_alphas({"E": 0.8, "A": -0.4})

    assert scaled == {"E": 0.4, "A": -0.4}


def test_build_language_backend_passes_configured_per_trait_strength():
    from orchestrator.cli import build_language_backend

    backend = build_language_backend(
        {
            "seed": 7,
            "steering": {
                "strength": 2.0,
                "per_trait_strength": {"E": 0.25},
            },
            "inference": {},
            "optimization": {},
        },
        {},
        {},
        mock=True,
    )

    assert backend._scale_alphas({"E": 1.0, "A": 1.0}) == {"E": 0.5, "A": 2.0}




def test_recommend_pareto_optimal_alphas_normal_case():
    dose_response = {
        "E": [
            {"alpha": 0.5, "trait_expression": 0.3, "coherence": 0.9},
            {"alpha": 1.0, "trait_expression": 0.7, "coherence": 0.8},
            {"alpha": 1.5, "trait_expression": 0.9, "coherence": 0.6},
        ],
    }
    result = recommend_pareto_optimal_alphas(dose_response, coherence_threshold=0.7)
    assert "E" in result
    rec = result["E"]
    assert rec.alpha == 1.0
    assert rec.trait_expression == 0.7
    assert rec.coherence == 0.8
    assert rec.threshold_met is True


def test_recommend_pareto_optimal_alphas_all_below_threshold():
    dose_response = {
        "N": [
            {"alpha": 0.5, "trait_expression": 0.3, "coherence": 0.4},
            {"alpha": 1.0, "trait_expression": 0.7, "coherence": 0.6},
            {"alpha": 1.5, "trait_expression": 0.9, "coherence": 0.5},
        ],
    }
    result = recommend_pareto_optimal_alphas(dose_response, coherence_threshold=0.7)
    assert "N" in result
    rec = result["N"]
    assert rec.alpha == 1.0
    assert rec.coherence == 0.6
    assert rec.threshold_met is False


def test_recommend_pareto_optimal_alphas_empty_input():
    result = recommend_pareto_optimal_alphas({})
    assert result == {}


def test_recommend_pareto_optimal_alphas_multiple_traits():
    dose_response = {
        "E": [
            {"alpha": 1.0, "trait_expression": 0.8, "coherence": 0.9},
            {"alpha": 2.0, "trait_expression": 0.95, "coherence": 0.65},
        ],
        "A": [
            {"alpha": 0.5, "trait_expression": 0.2, "coherence": 0.3},
            {"alpha": 1.0, "trait_expression": 0.5, "coherence": 0.5},
        ],
    }
    result = recommend_pareto_optimal_alphas(dose_response, coherence_threshold=0.7)
    assert result["E"].alpha == 1.0
    assert result["E"].threshold_met is True
    assert result["A"].alpha == 1.0
    assert result["A"].threshold_met is False

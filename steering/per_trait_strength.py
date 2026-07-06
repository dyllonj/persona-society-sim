"""Resolve per-trait steering strength multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


FALLBACK_KEYS = ("fallback", "default", "*")
TRAIT_ALIASES = {
    "extraversion": "E",
    "agreeableness": "A",
    "conscientiousness": "C",
    "openness": "O",
    "neuroticism": "N",
}


def _canonical_strength_key(key: str) -> str:
    stripped = key.strip()
    return TRAIT_ALIASES.get(stripped.lower(), stripped.upper())


def _resolve_fallback(per_trait_strength: Mapping[str, object]) -> float:
    fallback_names = {key.lower() for key in FALLBACK_KEYS}
    for key, value in per_trait_strength.items():
        if str(key).lower() in fallback_names:
            return float(value)
    return 1.0


def _normalize_strengths(per_trait_strength: Mapping[str, object]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    fallback_names = {key.lower() for key in FALLBACK_KEYS}
    for key, value in per_trait_strength.items():
        if key.lower() in fallback_names:
            continue
        normalized[_canonical_strength_key(str(key))] = float(value)
    return normalized


def resolve_per_trait_strength(
    raw_alphas: Mapping[str, float],
    per_trait_strength: Optional[Mapping[str, object]] = None,
    global_strength: float = 1.0,
) -> Dict[str, float]:
    """Map raw persona alphas to final steering alphas.

    The final alpha for each trait is:
        raw_alpha * global_strength * trait_strength

    Missing per-trait entries use the configured fallback strength, or 1.0
    when no fallback is configured.
    """

    strength_cfg = per_trait_strength or {}
    fallback_strength = _resolve_fallback(strength_cfg)
    trait_strengths = _normalize_strengths(strength_cfg)
    global_multiplier = float(global_strength)

    return {
        trait: float(alpha)
        * global_multiplier
        * trait_strengths.get(_canonical_strength_key(str(trait)), fallback_strength)
        for trait, alpha in raw_alphas.items()
    }


@dataclass
class TraitDoseResponse:
    """Dose-response data for a single trait across alpha values."""
    trait: str
    results: List[Dict[str, float]]


@dataclass
class ParetoOptimalAlpha:
    """Pareto-optimal alpha recommendation for a single trait."""
    alpha: float
    trait_expression: float
    coherence: float
    threshold_met: bool


def recommend_pareto_optimal_alphas(
    dose_response: Dict[str, List[Dict[str, float]]],
    coherence_threshold: float = 0.7,
) -> Dict[str, ParetoOptimalAlpha]:
    """Recommend the Pareto-optimal alpha per trait.

    Selects the alpha that maximizes trait_expression subject to coherence >= threshold.
    If no alpha meets the coherence threshold, selects the alpha with highest coherence.
    """
    recommendations: Dict[str, ParetoOptimalAlpha] = {}
    for trait, results in dose_response.items():
        if not results:
            continue
        passing = [r for r in results if r.get("coherence", 0.0) >= coherence_threshold]
        if passing:
            best = max(passing, key=lambda r: r.get("trait_expression", 0.0))
            recommendations[trait] = ParetoOptimalAlpha(
                alpha=best["alpha"],
                trait_expression=best.get("trait_expression", 0.0),
                coherence=best.get("coherence", 0.0),
                threshold_met=True,
            )
        else:
            best = max(results, key=lambda r: r.get("coherence", 0.0))
            recommendations[trait] = ParetoOptimalAlpha(
                alpha=best["alpha"],
                trait_expression=best.get("trait_expression", 0.0),
                coherence=best.get("coherence", 0.0),
                threshold_met=False,
            )
    return recommendations

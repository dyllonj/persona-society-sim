"""Trait band helpers for classifying agents into coarse cohorts."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

TRAIT_ORDER: Tuple[str, ...] = ("E", "A", "C", "O", "N", "TRUTH", "SYNC")
BAND_THRESHOLDS = {"low": -1.5, "high": 1.5}
BAND_METADATA = {
    "description": "Dominant trait bucket computed from persona coeffs + steering snapshot",
    "bands": {
        "low": {"max": BAND_THRESHOLDS["low"]},
        "neutral": {"range": [BAND_THRESHOLDS["low"], BAND_THRESHOLDS["high"]]},
        "high": {"min": BAND_THRESHOLDS["high"]},
    },
    "traits": TRAIT_ORDER,
}


def _combine_traits(persona_coeffs: Dict[str, float], steering_snapshot: Dict[str, float]) -> Dict[str, float]:
    combined = {}
    for trait in TRAIT_ORDER:
        base = persona_coeffs.get(trait, 0.0)
        delta = steering_snapshot.get(trait, 0.0)
        combined[trait] = base + delta
    return combined


def _band_for_value(value: float) -> str:
    if value <= BAND_THRESHOLDS["low"]:
        return "low"
    if value >= BAND_THRESHOLDS["high"]:
        return "high"
    return "neutral"


def trait_band_key(
    persona_coeffs: Dict[str, float],
    steering_snapshot: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Return a cohort key such as ``"E:high"`` for the dominant trait."""

    steering_snapshot = steering_snapshot or {}
    combined = _combine_traits(persona_coeffs, steering_snapshot)
    if not combined:
        return None
    dominant_trait = max(combined.items(), key=lambda item: abs(item[1]))
    trait, value = dominant_trait
    return f"{trait}:{_band_for_value(value)}"


def band_metadata(trait_key: Optional[str]) -> Dict[str, object]:
    if trait_key is None:
        return {}
    return BAND_METADATA

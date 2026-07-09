"""Macro-level measurement helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

from schemas.logs import MetricsSnapshot


def cooperation_rate(cooperation_events: List[bool]) -> float:
    """Share of eligible actions that were successful cooperative actions."""

    if not cooperation_events:
        return 0.0
    return sum(bool(event) for event in cooperation_events) / len(cooperation_events)


def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for idx, value in enumerate(sorted_vals, start=1):
        cumulative += value
        weighted_sum += idx * value
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def polarization(opinions: Dict[str, float]) -> float:
    if not opinions:
        return 0.0
    values = list(opinions.values())
    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / len(values)
    return variance


def build_metrics_snapshot(
    run_id: str,
    tick: int,
    cooperation_events: List[bool],
    wealth: Dict[str, float],
    opinions: Dict[str, float],
    conflicts: int,
    enforcement_cost: float,
    *,
    trait_key: Optional[str] = None,
    band_metadata: Optional[Dict[str, object]] = None,
    prompt_duplication_rate: float = 0.0,
    plan_reuse_rate: float = 0.0,
    action_type_entropy: float = 0.0,
    population_trait_variance: float = 0.0,
    variance_vs_mean_ratio: float = 0.0,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        run_id=run_id,
        tick=tick,
        cooperation_rate=cooperation_rate(cooperation_events),
        gini_wealth=gini(list(wealth.values())),
        polarization_modularity=polarization(opinions),
        conflicts=conflicts,
        rule_enforcement_cost=enforcement_cost,
        trait_key=trait_key,
        band_metadata=band_metadata or {},
        prompt_duplication_rate=prompt_duplication_rate,
        plan_reuse_rate=plan_reuse_rate,
        action_type_entropy=action_type_entropy,
        population_trait_variance=population_trait_variance,
        variance_vs_mean_ratio=variance_vs_mean_ratio,
    )

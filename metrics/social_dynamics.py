"""Macro-level measurement helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from schemas.logs import MetricsSnapshot


def cooperation_rate(task_outcomes: List[str]) -> float:
    if not task_outcomes:
        return 0.0
    successes = sum(1 for outcome in task_outcomes if outcome == "success")
    return successes / len(task_outcomes)


def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = np.sort(np.array(values))
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    return (n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n


def polarization(opinions: Dict[str, float]) -> float:
    if not opinions:
        return 0.0
    values = np.array(list(opinions.values()))
    return float(values.var())


def build_metrics_snapshot(
    run_id: str,
    tick: int,
    cooperation_events: List[str],
    wealth: Dict[str, float],
    opinions: Dict[str, float],
    conflicts: int,
    enforcement_cost: float,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        run_id=run_id,
        tick=tick,
        cooperation_rate=cooperation_rate(cooperation_events),
        gini_wealth=gini(list(wealth.values())),
        polarization_modularity=polarization(opinions),
        conflicts=conflicts,
        rule_enforcement_cost=enforcement_cost,
    )

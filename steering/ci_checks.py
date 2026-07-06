"""Lightweight validation utilities for steering CI sweeps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import math


@dataclass(frozen=True)
class TraitCurvePoint:
    trait: str
    seed: int
    alpha: float
    logprob_gap_delta: float
    source: Path


@dataclass(frozen=True)
class TraitDirectionality:
    trait: str
    seed: int
    alpha: float
    sign_consistency: float
    directional_improvement: float
    source: Path


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Return cosine similarity with zero-norm protection."""

    dot_product = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(float(b) ** 2 for b in vec_b))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(dot_product / denom)


def validate_cosine_stability(
    vectors: Mapping[str, Mapping[int, Sequence[Tuple[int, Sequence[float]]]]],
    threshold: float,
) -> List[str]:
    """Ensure steering vectors stay consistent across seeds for every layer."""

    failures: List[str] = []
    for trait, layer_map in vectors.items():
        for layer, seed_vectors in layer_map.items():
            if len(seed_vectors) < 2:
                continue
            base_seed, base_vec = seed_vectors[0]
            for seed, candidate in seed_vectors[1:]:
                cos = cosine_similarity(base_vec, candidate)
                if cos < threshold:
                    failures.append(
                        (
                            f"trait={trait} layer={layer} seed_pair=({base_seed},{seed}) "
                            f"cosine={cos:.4f} < {threshold:.4f}"
                        )
                    )
    return failures


def validate_directionality(
    points: Iterable[TraitDirectionality],
    *,
    sign_threshold: float,
    directional_threshold: float,
) -> List[str]:
    """Gate runs on sign-consistency and directional-improvement metrics."""

    failures: List[str] = []
    for point in points:
        if point.sign_consistency < sign_threshold:
            failures.append(
                (
                    f"{point.trait} seed={point.seed} alpha={point.alpha} "
                    f"sign_consistency={point.sign_consistency:.3f} < {sign_threshold:.3f}"
                )
            )
        if point.directional_improvement < directional_threshold:
            failures.append(
                (
                    f"{point.trait} seed={point.seed} alpha={point.alpha} "
                    f"directional_improvement={point.directional_improvement:.3f} < "
                    f"{directional_threshold:.3f}"
                )
            )
    return failures


def validate_anti_steerable_fraction(
    trait_rows: Iterable[Mapping[str, object]],
    *,
    threshold: float = 0.5,
) -> List[str]:
    """Fail traits whose anti-steerable fraction is too high."""

    failures: List[str] = []
    for row in trait_rows:
        value = float(row.get("anti_steerable_fraction", 0.0) or 0.0)
        if value > threshold:
            trait = row.get("trait_name") or row.get("trait") or "unknown"
            failures.append(
                f"{trait} anti_steerable_fraction={value:.3f} > {threshold:.3f}"
            )
    return failures


def validate_directional_agreement_metadata(
    metadata: Mapping[str, object],
    *,
    threshold: float = 0.3,
) -> List[str]:
    """Fail vector metadata whose per-layer directional agreement is weak."""

    failures: List[str] = []
    trait = metadata.get("trait") or metadata.get("vector_store_id") or "unknown"
    for layer in metadata.get("layers", []) or []:
        if not isinstance(layer, Mapping):
            continue
        if "directional_agreement" not in layer:
            continue
        value = float(layer.get("directional_agreement") or 0.0)
        if value < threshold:
            failures.append(
                f"{trait} layer={layer.get('layer_id')} directional_agreement="
                f"{value:.3f} < {threshold:.3f}"
            )
    return failures


def validate_bleed_matrix(
    bleed_matrix: Mapping[str, Mapping[str, float]],
    *,
    threshold: float = 2.0,
) -> List[str]:
    """Flag excessive off-diagonal cross-trait bleed."""

    failures: List[str] = []
    for source, row in bleed_matrix.items():
        for target, value in row.items():
            if source == target:
                continue
            numeric = abs(float(value))
            if numeric > threshold:
                failures.append(
                    f"bleed {source}->{target}={numeric:.3f} > {threshold:.3f}"
                )
    return failures


def validate_monotonic_logprobs(
    points: Iterable[TraitCurvePoint], *, tolerance: float
) -> List[str]:
    """Assert that log-prob deltas grow monotonically with alpha per seed."""

    grouped: Dict[Tuple[str, int], List[TraitCurvePoint]] = defaultdict(list)
    for point in points:
        grouped[(point.trait, point.seed)].append(point)

    failures: List[str] = []
    for (trait, seed), entries in grouped.items():
        ordered = sorted(entries, key=lambda item: item.alpha)
        for previous, current in zip(ordered, ordered[1:]):
            if current.logprob_gap_delta + tolerance < previous.logprob_gap_delta:
                failures.append(
                    (
                        f"{trait} seed={seed} alpha={previous.alpha}->{current.alpha} "
                        f"logprob delta {previous.logprob_gap_delta:.4f}->{current.logprob_gap_delta:.4f} "
                        f"(tolerance {tolerance})"
                    )
                )
    return failures


def extract_trait_rows(report: dict) -> List[dict]:
    """Return trait rows from a steering evaluation report payload."""

    return list(report.get("traits", []))

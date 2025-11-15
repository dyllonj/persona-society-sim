"""Aggregation helpers for structured research telemetry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Set

from schemas.logs import CitationLog, ReportGradeLog, ResearchFactLog


@dataclass
class FactStats:
    fact_ids: Set[str] = field(default_factory=set)
    correct_fact_ids: Set[str] = field(default_factory=set)


@dataclass
class CitationStats:
    total: int = 0
    doc_ids: Set[str] = field(default_factory=set)


@dataclass
class GradeStats:
    rewards: List[float] = field(default_factory=list)


class ResearchMetricAggregator:
    """Track research sprint telemetry by persona trait cohorts."""

    def __init__(self) -> None:
        self.targets_total: int = 0
        self.fact_stats: DefaultDict[str, FactStats] = defaultdict(FactStats)
        self.citation_stats: DefaultDict[str, CitationStats] = defaultdict(CitationStats)
        self.grade_stats: DefaultDict[str, GradeStats] = defaultdict(GradeStats)

    def observe_fact(self, log: ResearchFactLog) -> None:
        for key in self._cohort_keys(log.trait_key):
            stats = self.fact_stats[key]
            stats.fact_ids.add(log.fact_id)
            if log.correct:
                stats.correct_fact_ids.add(log.fact_id)

    def observe_citation(self, log: CitationLog) -> None:
        for key in self._cohort_keys(log.trait_key):
            stats = self.citation_stats[key]
            stats.total += 1
            if log.doc_id:
                stats.doc_ids.add(log.doc_id)

    def observe_grade(self, log: ReportGradeLog) -> None:
        self.targets_total = max(self.targets_total, log.targets_total)
        for key in self._cohort_keys(log.trait_key):
            stats = self.grade_stats[key]
            stats.rewards.append(log.reward_points)

    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        coverage: Dict[str, Dict[str, float]] = {}
        for key, stats in self.fact_stats.items():
            coverage[key] = {
                "facts_observed": float(len(stats.fact_ids)),
                "facts_correct": float(len(stats.correct_fact_ids)),
                "coverage": self._coverage_ratio(len(stats.correct_fact_ids)),
            }
        citations: Dict[str, Dict[str, float]] = {}
        for key, stats in self.citation_stats.items():
            diversity = (len(stats.doc_ids) / stats.total) if stats.total else 0.0
            citations[key] = {
                "total": float(stats.total),
                "unique_docs": float(len(stats.doc_ids)),
                "diversity": round(diversity, 3),
            }
        grade: Dict[str, Dict[str, float]] = {}
        global_avg = self._avg_reward(self.grade_stats.get("global"))
        for key, stats in self.grade_stats.items():
            avg = self._avg_reward(stats)
            drift = avg - global_avg if global_avg is not None else 0.0
            grade[key] = {
                "avg_reward": round(avg, 3) if avg is not None else 0.0,
                "drift": round(drift, 3) if avg is not None else 0.0,
            }
        return {
            "fact_coverage": coverage,
            "citation_diversity": citations,
            "grade_drift": grade,
            "targets_total": self.targets_total,
        }

    def _cohort_keys(self, trait_key: str | None) -> List[str]:
        keys = ["global"]
        if trait_key and trait_key not in keys:
            keys.append(trait_key)
        return keys

    def _coverage_ratio(self, correct: int) -> float:
        if not self.targets_total:
            return 0.0
        return round(correct / self.targets_total, 3)

    @staticmethod
    def _avg_reward(stats: GradeStats | None) -> float | None:
        if not stats or not stats.rewards:
            return None
        return sum(stats.rewards) / len(stats.rewards)


__all__ = ["ResearchMetricAggregator"]

"""Rubric-based scorers for behavioral probes and organic actions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from schemas.logs import ActionLog, BehaviorProbeLog


@dataclass
class RubricScore:
    trait: str
    cohort: str
    trait_band: Optional[str]
    source: str
    affordance: str
    value: float
    details: Dict[str, object] = field(default_factory=dict)


class ProbeRubricScorer:
    """Emit per-trait rubric scores from ActionLog and BehaviorProbeLog entries."""

    # action_type -> (trait, affordance, weight)
    ACTION_AFFORDANCES: Dict[str, Tuple[str, str, float]] = {
        "gift": ("A", "generosity", 1.0),
        "work": ("C", "diligence", 1.0),
        "talk": ("E", "outreach", 0.5),
        "move": ("O", "exploration", 0.25),
        "report": ("N", "vigilance", 1.0),
        "enforce": ("N", "vigilance", 0.5),
    }

    def __init__(
        self,
        behavior_targets: Optional[Dict[str, Tuple[str, Optional[str], Optional[str]]]] = None,
    ) -> None:
        # probe_id -> (trait, affordance, preferred_outcome)
        self.behavior_targets: Dict[str, Tuple[str, Optional[str], Optional[str]]] = (
            behavior_targets or {}
        )
        self.scores: List[RubricScore] = []
        self._visited_locations: DefaultDict[str, Set[str]] = defaultdict(set)

    def ingest_actions(self, logs: Iterable[ActionLog]) -> None:
        for log in logs:
            self.ingest_action(log)

    def ingest_action(self, log: ActionLog) -> None:
        trait, trait_key, trait_band = self._trait_fields_from_action(log)
        if not trait:
            return
        affordance_def = self.ACTION_AFFORDANCES.get(log.action_type)
        if not affordance_def or log.outcome != "success":
            return
        affordance_trait, affordance, weight = affordance_def
        if affordance_trait != trait:
            return
        details: Dict[str, object] = {
            "action_type": log.action_type,
            "outcome": log.outcome,
        }
        if log.action_type == "move":
            destination = self._destination_from_params(log.params)
            if not destination:
                return
            is_new_destination = destination not in self._visited_locations[log.agent_id]
            self._visited_locations[log.agent_id].add(destination)
            if not is_new_destination:
                return
            details.update({"destination": destination, "novel_destination": True})
        self._record_score(trait, trait_key, trait_band, "action", affordance, weight, details)

    def ingest_behavior_probes(self, logs: Iterable[BehaviorProbeLog]) -> None:
        for log in logs:
            self.ingest_behavior_probe(log)

    def ingest_behavior_probe(self, log: BehaviorProbeLog) -> None:
        target_trait, target_affordance, preferred_outcome = self._behavior_target(log.probe_id)
        trait = log.trait or target_trait
        affordance = log.affordance or target_affordance or "probe_response"
        if not trait:
            return
        preferred = log.preferred_outcome or preferred_outcome
        value = 0.0
        if log.outcome:
            if preferred:
                value = 1.0 if log.outcome == preferred else -1.0
            else:
                value = 1.0
        details = {
            "probe_id": log.probe_id,
            "outcome": log.outcome,
            "preferred_outcome": preferred,
        }
        cohort = log.trait_key or self._cohort_from_parts(trait, log.trait_band)
        self._record_score(trait, cohort, log.trait_band, "probe", affordance, value, details)

    def scores_by_trait(self) -> Dict[str, Dict[str, float]]:
        totals: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
        for score in self.scores:
            bucket = totals[score.trait]
            bucket[score.affordance] += score.value
            bucket[f"{score.source}_total"] += score.value
        return {trait: dict(values) for trait, values in totals.items()}

    def cohort_breakdown(self) -> Dict[str, Dict[str, float]]:
        totals: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
        for score in self.scores:
            bucket = totals[score.cohort]
            bucket[score.affordance] += score.value
            bucket[f"{score.source}_total"] += score.value
        return {cohort: dict(values) for cohort, values in totals.items()}

    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            "by_trait": self.scores_by_trait(),
            "by_cohort": self.cohort_breakdown(),
        }

    def _record_score(
        self,
        trait: str,
        trait_key: Optional[str],
        trait_band: Optional[str],
        source: str,
        affordance: str,
        value: float,
        details: Dict[str, object],
    ) -> None:
        cohort = trait_key or self._cohort_from_parts(trait, trait_band)
        self.scores.append(
            RubricScore(
                trait=trait,
                cohort=cohort,
                trait_band=trait_band,
                source=source,
                affordance=affordance,
                value=value,
                details=details,
            )
        )

    def _cohort_from_parts(self, trait: str, trait_band: Optional[str]) -> str:
        return f"{trait}:{trait_band}" if trait_band else trait

    def _behavior_target(self, probe_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return self.behavior_targets.get(probe_id, (None, None, None))

    def _trait_fields_from_action(
        self, log: ActionLog
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        info = log.info or {}
        trait_key = info.get("trait_key")
        trait_band = info.get("trait_band")
        trait_name: Optional[str] = None
        if trait_key:
            if ":" in trait_key:
                trait_name, key_band = trait_key.split(":", 1)
                trait_band = trait_band or key_band
            else:
                trait_name = trait_key
        return trait_name, trait_key, trait_band

    @staticmethod
    def _destination_from_params(params: Dict[str, str]) -> Optional[str]:
        return params.get("destination") or params.get("room_id")


__all__ = ["ProbeRubricScorer", "RubricScore"]

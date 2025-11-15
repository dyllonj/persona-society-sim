"""Probe scheduling utilities for Likert and behavioral interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import re
from typing import Dict, List, Literal, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

DEFAULT_PROBE_PATH = Path("configs/probes.yaml")
LikertScore = Optional[int]


@dataclass
class LikertProbeDefinition:
    probe_id: str
    question: str
    trait: Optional[str]
    instructions: str
    cadence: int


@dataclass
class BehaviorProbeDefinition:
    probe_id: str
    scenario: str
    instructions: str
    outcomes: Dict[str, List[str]]
    cadence: int


@dataclass
class ProbeAssignment:
    probe_id: str
    kind: Literal["likert", "behavior"]
    prompt: str
    scheduled_tick: int
    question: Optional[str] = None
    trait: Optional[str] = None
    scenario: Optional[str] = None
    outcomes: Dict[str, List[str]] = field(default_factory=dict)
    cooldown: int = 0

    def inject(self, observation: str) -> str:
        prefix = ["[Probe] You have been selected for a research probe.", self.prompt]
        if self.kind == "likert" and self.question:
            prefix.append(f"Question: {self.question}")
        if self.kind == "behavior" and self.scenario:
            prefix.append(f"Scenario: {self.scenario}")
        prefix.append("Provide your probe response before continuing with normal conversation.")
        joined = "\n".join(prefix).strip()
        return f"{joined}\n\nOriginal observation:\n{observation}".strip()


class ProbeManager:
    """Assigns probe prompts to agents and parses their responses."""

    _LIKERT_REGEX = re.compile(r"\b([1-5])\b")
    _LIKERT_KEYWORDS: List[Tuple[int, Tuple[str, ...]]] = [
        (5, ("strongly agree", "completely agree", "absolutely", "definitely")),
        (1, ("strongly disagree", "absolutely not", "never", "refuse")),
        (4, ("agree", "support", "probably", "likely")),
        (2, ("disagree", "doubt", "probably not", "unlikely")),
        (3, ("neutral", "unsure", "mixed", "uncertain")),
    ]

    def __init__(
        self,
        likert_probes: List[LikertProbeDefinition],
        behavior_probes: List[BehaviorProbeDefinition],
        *,
        likert_interval: int = 30,
        behavior_interval: int = 45,
        seed: int = 7,
    ):
        self.likert_probes = list(likert_probes)
        self.behavior_probes = list(behavior_probes)
        self.likert_interval = likert_interval
        self.behavior_interval = behavior_interval
        self.random = random.Random(seed)
        self._active: Dict[str, ProbeAssignment] = {}
        self._next_due: Dict[str, Dict[str, int]] = {
            "likert": {},
            "behavior": {},
        }

    @classmethod
    def from_config(cls, config: Optional[Dict]) -> Optional["ProbeManager"]:
        if not config or not config.get("enabled", True):
            return None
        if yaml is None:
            return None
        path = Path(config.get("definitions_path", DEFAULT_PROBE_PATH))
        if not path.exists():
            return None
        payload = yaml.safe_load(path.read_text()) or {}
        likert_cfg = payload.get("likert", {}) or {}
        behavior_cfg = payload.get("behavior", {}) or {}
        likert_interval = int(config.get("likert_cadence", likert_cfg.get("cadence", 30)))
        behavior_interval = int(config.get("behavior_cadence", behavior_cfg.get("cadence", 45)))
        likert_defs: List[LikertProbeDefinition] = []
        for entry in likert_cfg.get("questions", []):
            cadence = int(entry.get("cadence", likert_interval))
            likert_defs.append(
                LikertProbeDefinition(
                    probe_id=str(entry.get("id")),
                    question=str(entry.get("question", "")),
                    trait=entry.get("trait"),
                    instructions=str(entry.get("instructions", "Respond with 1-5.")),
                    cadence=cadence,
                )
            )
        behavior_defs: List[BehaviorProbeDefinition] = []
        for entry in behavior_cfg.get("scenarios", []):
            cadence = int(entry.get("cadence", behavior_interval))
            outcomes_cfg = entry.get("outcomes", {}) or {}
            outcomes: Dict[str, List[str]] = {}
            for label, label_cfg in outcomes_cfg.items():
                keywords = label_cfg.get("keywords", []) if isinstance(label_cfg, dict) else label_cfg
                if isinstance(keywords, str):
                    keywords = [keywords]
                outcomes[label] = [kw.lower() for kw in keywords]
            behavior_defs.append(
                BehaviorProbeDefinition(
                    probe_id=str(entry.get("id")),
                    scenario=str(entry.get("scenario", "")),
                    instructions=str(entry.get("instructions", "Describe what you would do.")),
                    outcomes=outcomes,
                    cadence=cadence,
                )
            )
        if not likert_defs and not behavior_defs:
            return None
        return cls(
            likert_defs,
            behavior_defs,
            likert_interval=likert_interval,
            behavior_interval=behavior_interval,
            seed=int(config.get("seed", 7)),
        )

    def assign_probe(self, agent_id: str, tick: int) -> Optional[ProbeAssignment]:
        if agent_id in self._active:
            return self._active[agent_id]
        due_kinds = [
            kind
            for kind, definitions in (
                ("likert", self.likert_probes),
                ("behavior", self.behavior_probes),
            )
            if definitions and tick >= self._next_due[kind].get(agent_id, 0)
        ]
        if not due_kinds:
            return None
        # Prioritize Likert probes for stability, fall back to behavior
        due_kinds.sort(
            key=lambda k: (self._next_due[k].get(agent_id, 0), 0 if k == "likert" else 1)
        )
        kind = due_kinds[0]
        assignment = self._build_assignment(kind, tick)
        if assignment:
            self._active[agent_id] = assignment
        return assignment

    def pending_probe(self, agent_id: str, tick: int) -> Optional[ProbeAssignment]:
        assignment = self._active.get(agent_id)
        if assignment and tick >= assignment.scheduled_tick:
            return assignment
        return None

    def complete_probe(
        self,
        agent_id: str,
        assignment: Optional[ProbeAssignment],
        tick: int,
    ) -> None:
        self._active.pop(agent_id, None)
        if not assignment:
            return
        kind = assignment.kind
        interval = assignment.cooldown or (
            self.likert_interval if kind == "likert" else self.behavior_interval
        )
        self._next_due[kind][agent_id] = tick + interval

    def _build_assignment(self, kind: str, tick: int) -> Optional[ProbeAssignment]:
        if kind == "likert" and self.likert_probes:
            definition = self.random.choice(self.likert_probes)
            prompt = f"{definition.instructions.strip()}"
            return ProbeAssignment(
                probe_id=definition.probe_id,
                kind="likert",
                prompt=prompt,
                scheduled_tick=tick,
                question=definition.question,
                trait=definition.trait,
                cooldown=definition.cadence,
            )
        if kind == "behavior" and self.behavior_probes:
            definition = self.random.choice(self.behavior_probes)
            prompt = f"{definition.instructions.strip()}"
            outcomes = {
                label: [kw.lower() for kw in keywords]
                for label, keywords in (definition.outcomes or {}).items()
            }
            return ProbeAssignment(
                probe_id=definition.probe_id,
                kind="behavior",
                prompt=prompt,
                scheduled_tick=tick,
                scenario=definition.scenario,
                outcomes=outcomes,
                cooldown=definition.cadence,
            )
        return None

    @classmethod
    def score_likert_response(cls, response_text: str) -> Tuple[LikertScore, str]:
        text = response_text.strip()
        match = cls._LIKERT_REGEX.search(text)
        if match:
            return int(match.group(1)), "numeric"
        lowered = text.lower()
        for score, keywords in cls._LIKERT_KEYWORDS:
            for keyword in keywords:
                if keyword in lowered:
                    return score, keyword
        return None, "unparsed"

    @staticmethod
    def score_behavior_response(assignment: ProbeAssignment, response_text: str) -> Tuple[Optional[str], str]:
        lowered = response_text.lower()
        for label, keywords in assignment.outcomes.items():
            for keyword in keywords:
                if keyword and keyword in lowered:
                    return label, keyword
            if label.lower() in lowered:
                return label, label
        return None, "unparsed"

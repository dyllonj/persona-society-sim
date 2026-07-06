"""Probe scheduling utilities for Likert and behavioral interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import re
from typing import Dict, List, Literal, Optional, Tuple

from schemas.logs import PersonaStabilityLog

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

DEFAULT_PROBE_PATH = Path("configs/probes.yaml")
LikertScore = Optional[int]
ProbeKind = Literal["likert", "behavior", "persona_stability"]

DEFAULT_PERSONA_STABILITY_KEYWORDS: Dict[str, List[str]] = {
    "E": ["talk", "social", "group", "people", "outgoing", "meet"],
    "A": ["help", "support", "kind", "cooperate", "share", "agree"],
    "C": ["plan", "task", "careful", "schedule", "complete", "reliable"],
    "O": ["learn", "explore", "novel", "curious", "ideas", "creative"],
    "N": ["worry", "stress", "nervous", "uncertain", "calm", "steady"],
}


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
    trait: Optional[str] = None
    affordance: Optional[str] = None
    preferred_outcome: Optional[str] = None


@dataclass
class PersonaStabilityProbeDefinition:
    probe_id: str
    prompt: str
    cadence: int = 20
    trait_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            trait: list(keywords)
            for trait, keywords in DEFAULT_PERSONA_STABILITY_KEYWORDS.items()
        }
    )


@dataclass
class ProbeAssignment:
    probe_id: str
    kind: ProbeKind
    prompt: str
    scheduled_tick: int
    question: Optional[str] = None
    trait: Optional[str] = None
    scenario: Optional[str] = None
    outcomes: Dict[str, List[str]] = field(default_factory=dict)
    cooldown: int = 0
    affordance: Optional[str] = None
    preferred_outcome: Optional[str] = None
    trait_keywords: Dict[str, List[str]] = field(default_factory=dict)

    def inject(self, observation: str) -> str:
        prefix = ["[Probe] You have been selected for a research probe.", self.prompt]
        if self.kind == "likert" and self.question:
            prefix.append(f"Question: {self.question}")
        if self.kind == "behavior" and self.scenario:
            prefix.append(f"Scenario: {self.scenario}")
        if self.kind == "persona_stability":
            prefix.append("Describe your current persona, priorities, and typical behavior.")
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
        persona_stability_probes: Optional[List[PersonaStabilityProbeDefinition]] = None,
        *,
        likert_interval: int = 30,
        behavior_interval: int = 45,
        persona_stability_interval: int = 20,
        seed: int = 7,
    ):
        self.likert_probes = list(likert_probes)
        self.behavior_probes = list(behavior_probes)
        self.persona_stability_probes = list(persona_stability_probes or [])
        self.likert_interval = likert_interval
        self.behavior_interval = behavior_interval
        self.persona_stability_interval = persona_stability_interval
        self.random = random.Random(seed)
        self._active: Dict[str, ProbeAssignment] = {}
        self._next_due: Dict[str, Dict[str, int]] = {
            "likert": {},
            "behavior": {},
            "persona_stability": {},
        }
        self._persona_baselines: Dict[Tuple[str, str], str] = {}
        self._validate_behavior_affordances()

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
        persona_cfg = payload.get("persona_stability", {}) or {}
        likert_interval = int(config.get("likert_cadence", likert_cfg.get("cadence", 30)))
        behavior_interval = int(config.get("behavior_cadence", behavior_cfg.get("cadence", 45)))
        persona_interval = int(
            config.get("persona_stability_cadence", persona_cfg.get("cadence", 20))
        )
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
                    trait=entry.get("trait"),
                    affordance=entry.get("affordance"),
                    scenario=str(entry.get("scenario", "")),
                    instructions=str(entry.get("instructions", "Describe what you would do.")),
                    outcomes=outcomes,
                    cadence=cadence,
                    preferred_outcome=entry.get("preferred_outcome"),
                )
            )
        persona_defs: List[PersonaStabilityProbeDefinition] = []
        if persona_cfg.get("enabled", False):
            prompt_entries = persona_cfg.get("prompts") or [
                {
                    "id": "persona_stability",
                    "prompt": "Briefly describe who you are in this simulation and how you usually decide what to do.",
                }
            ]
            for entry in prompt_entries:
                trait_keywords = entry.get("trait_keywords", DEFAULT_PERSONA_STABILITY_KEYWORDS)
                persona_defs.append(
                    PersonaStabilityProbeDefinition(
                        probe_id=str(entry.get("id", "persona_stability")),
                        prompt=str(
                            entry.get(
                                "prompt",
                                "Briefly describe who you are in this simulation and how you usually decide what to do.",
                            )
                        ),
                        cadence=int(entry.get("cadence", persona_interval)),
                        trait_keywords={
                            trait: [str(keyword).lower() for keyword in keywords]
                            for trait, keywords in trait_keywords.items()
                        },
                    )
                )
        if not likert_defs and not behavior_defs and not persona_defs:
            return None
        return cls(
            likert_defs,
            behavior_defs,
            persona_defs,
            likert_interval=likert_interval,
            behavior_interval=behavior_interval,
            persona_stability_interval=persona_interval,
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
                ("persona_stability", self.persona_stability_probes),
            )
            if definitions and tick >= self._next_due[kind].get(agent_id, 0)
        ]
        if not due_kinds:
            return None
        # Prioritize Likert probes for stability, fall back to behavior
        priority = {"persona_stability": 0, "likert": 1, "behavior": 2}
        due_kinds.sort(
            key=lambda k: (self._next_due[k].get(agent_id, 0), priority.get(k, 99))
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
            self.likert_interval
            if kind == "likert"
            else self.behavior_interval
            if kind == "behavior"
            else self.persona_stability_interval
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
                trait=definition.trait,
                affordance=definition.affordance,
                preferred_outcome=definition.preferred_outcome,
            )
        if kind == "persona_stability" and self.persona_stability_probes:
            definition = self.random.choice(self.persona_stability_probes)
            return ProbeAssignment(
                probe_id=definition.probe_id,
                kind="persona_stability",
                prompt=definition.prompt.strip(),
                scheduled_tick=tick,
                cooldown=definition.cadence,
                trait_keywords={
                    trait: [kw.lower() for kw in keywords]
                    for trait, keywords in definition.trait_keywords.items()
                },
            )
        return None

    def _validate_behavior_affordances(self) -> None:
        """Ensure each behavior probe maps to a single, stable affordance."""

        affordance_owners: Dict[str, str] = {}
        for probe in self.behavior_probes:
            if not probe.affordance:
                raise ValueError(f"Probe {probe.probe_id} is missing an affordance tag")
            if not probe.trait:
                raise ValueError(f"Probe {probe.probe_id} must declare a dominant trait")
            owner = affordance_owners.setdefault(probe.affordance, probe.trait)
            if owner != probe.trait:
                raise ValueError(
                    f"Affordance {probe.affordance} is shared by multiple traits: {owner} vs {probe.trait}"
                )
            if probe.preferred_outcome and probe.preferred_outcome not in probe.outcomes:
                raise ValueError(
                    f"Probe {probe.probe_id} preferred_outcome={probe.preferred_outcome} is missing from outcomes"
                )

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

    def record_persona_stability_response(
        self,
        agent_id: str,
        assignment: ProbeAssignment,
        tick: int,
        response_text: str,
    ) -> PersonaStabilityLog:
        baseline_key = (agent_id, assignment.probe_id)
        baseline = self._persona_baselines.get(baseline_key)
        if baseline is None:
            baseline = response_text
            self._persona_baselines[baseline_key] = response_text
        distance = self.keyword_overlap_distance(baseline, response_text)
        return PersonaStabilityLog(
            agent_id=agent_id,
            tick=tick,
            probe_text=assignment.prompt,
            embedding_distance_from_baseline=distance,
            trait_scores=self.score_persona_trait_keywords(
                response_text,
                assignment.trait_keywords or DEFAULT_PERSONA_STABILITY_KEYWORDS,
            ),
        )

    @classmethod
    def score_persona_trait_keywords(
        cls,
        response_text: str,
        trait_keywords: Dict[str, List[str]],
    ) -> Dict[str, float]:
        tokens = cls._keyword_tokens(response_text)
        scores: Dict[str, float] = {}
        for trait, keywords in trait_keywords.items():
            keyword_set = {keyword.lower() for keyword in keywords if keyword}
            if not keyword_set:
                continue
            matches = len(tokens.intersection(keyword_set))
            scores[trait] = matches / len(keyword_set)
        return scores

    @classmethod
    def keyword_overlap_distance(cls, baseline_text: str, response_text: str) -> float:
        baseline_tokens = cls._keyword_tokens(baseline_text)
        response_tokens = cls._keyword_tokens(response_text)
        if not baseline_tokens and not response_tokens:
            return 0.0
        union = baseline_tokens.union(response_tokens)
        if not union:
            return 0.0
        similarity = len(baseline_tokens.intersection(response_tokens)) / len(union)
        return round(1.0 - similarity, 6)

    @staticmethod
    def _keyword_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z][a-z_'-]*", text.lower())
            if len(token) >= 3
        }

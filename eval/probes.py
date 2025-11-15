"""Scheduling and scoring helpers for evaluation probes."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Literal, Optional, Sequence

from metrics.social_dynamics import evaluate_behavior_rubric
from schemas.logs import ActionLog

ProbeType = Literal["self_report", "behavioral"]
InjectionMode = Literal["observation_tag", "encounter"]
ResponseMode = Literal["utterance", "action"]


@dataclass
class QuestionnaireItem:
    item_id: str
    trait: str
    text: str
    reverse_scored: bool = False


@dataclass
class ProbeDefinition:
    probe_id: str
    probe_type: ProbeType
    start_tick: int
    cadence: int
    prompt: str
    injection_mode: InjectionMode = "observation_tag"
    response_mode: ResponseMode = "utterance"
    questionnaire: List[QuestionnaireItem] = field(default_factory=list)
    rubric: Optional[str] = None
    targets: Optional[List[str]] = None

    def prompt_block(self) -> str:
        if self.probe_type == "self_report" and self.questionnaire:
            header = self.prompt or DEFAULT_SELF_REPORT_PROMPT
            questions = "\n".join(
                f"{idx + 1}. {item.text} (1-5)"
                for idx, item in enumerate(self.questionnaire)
            )
            return f"{header}\n{questions}"
        return self.prompt


@dataclass
class ActiveProbe:
    probe_id: str
    probe_type: ProbeType
    agent_id: str
    prompt: str
    injection_mode: InjectionMode
    response_mode: ResponseMode
    questionnaire: List[QuestionnaireItem] = field(default_factory=list)
    rubric: Optional[str] = None


DEFAULT_SELF_REPORT_PROMPT = (
    "Self-report probe: rate each statement from 1 (disagree) to 5 (agree)."
)
DEFAULT_BEHAVIOR_PROMPT = (
    "Behavioral probe: a neighbor requests help with a small civic task."
)
DEFAULT_QUESTIONNAIRE: List[QuestionnaireItem] = [
    QuestionnaireItem("ipip_e1", "E", "I am talkative."),
    QuestionnaireItem("ipip_a1", "A", "I sympathize with others' feelings."),
    QuestionnaireItem("ipip_c1", "C", "I get chores done right away."),
]
LIKERT_PATTERN = re.compile(r"([1-5])")


class ProbeManager:
    """Creates probe prompts and scores responses."""

    def __init__(self, agent_ids: Iterable[str], config: Optional[Dict] = None):
        self.agent_ids = sorted(set(agent_ids))
        self.definitions = self._parse_config(config or {})
        self.enabled = bool(self.definitions)

    def tick(self, tick: int) -> List[ActiveProbe]:
        if not self.enabled:
            return []
        probes: List[ActiveProbe] = []
        for definition in self.definitions:
            if tick < definition.start_tick:
                continue
            cadence = max(1, definition.cadence)
            if (tick - definition.start_tick) % cadence != 0:
                continue
            targets = definition.targets or self.agent_ids
            valid_targets = [agent for agent in targets if agent in self.agent_ids]
            for agent_id in valid_targets:
                probes.append(
                    ActiveProbe(
                        probe_id=definition.probe_id,
                        probe_type=definition.probe_type,
                        agent_id=agent_id,
                        prompt=definition.prompt_block(),
                        injection_mode=definition.injection_mode,
                        response_mode=definition.response_mode,
                        questionnaire=list(definition.questionnaire),
                        rubric=definition.rubric,
                    )
                )
        return probes

    def score_probe(
        self,
        probe: ActiveProbe,
        *,
        raw_text: Optional[str] = None,
        action_log: Optional[ActionLog] = None,
    ) -> Dict[str, float]:
        if probe.probe_type == "self_report":
            return parse_likert_responses(raw_text or "", probe.questionnaire)
        if probe.probe_type == "behavioral" and action_log is not None:
            rubric = probe.rubric or "cooperation"
            return evaluate_behavior_rubric(action_log, rubric)
        return {}

    def _parse_config(self, config: Dict) -> List[ProbeDefinition]:
        if not config or not config.get("enabled", False):
            return []
        definitions: List[ProbeDefinition] = []
        questionnaires = config.get("questionnaires") or []
        if isinstance(questionnaires, dict):
            questionnaires = [questionnaires]
        for entry in questionnaires:
            questions = self._build_questionnaire(entry.get("questions"))
            prompt = entry.get("prompt") or DEFAULT_SELF_REPORT_PROMPT
            definitions.append(
                ProbeDefinition(
                    probe_id=entry.get("probe_id", "self_report"),
                    probe_type="self_report",
                    start_tick=int(entry.get("start_tick", 0)),
                    cadence=int(entry.get("cadence", 5)),
                    prompt=prompt,
                    injection_mode=entry.get("injection_mode", "observation_tag"),
                    response_mode="utterance",
                    questionnaire=questions,
                    targets=entry.get("targets"),
                )
            )

        scenarios = config.get("scenarios") or []
        if isinstance(scenarios, dict):
            scenarios = [scenarios]
        for entry in scenarios:
            prompt = entry.get("prompt") or DEFAULT_BEHAVIOR_PROMPT
            definitions.append(
                ProbeDefinition(
                    probe_id=entry.get("probe_id", "behavioral"),
                    probe_type="behavioral",
                    start_tick=int(entry.get("start_tick", 0)),
                    cadence=int(entry.get("cadence", 10)),
                    prompt=prompt,
                    injection_mode=entry.get("injection_mode", "encounter"),
                    response_mode="action",
                    rubric=entry.get("rubric", "cooperation"),
                    targets=entry.get("targets"),
                )
            )

        return definitions

    def _build_questionnaire(
        self, questions: Optional[Sequence[Dict[str, object]]]
    ) -> List[QuestionnaireItem]:
        if not questions:
            return list(DEFAULT_QUESTIONNAIRE)
        built: List[QuestionnaireItem] = []
        for entry in questions:
            built.append(
                QuestionnaireItem(
                    item_id=str(entry.get("item_id", entry.get("trait", "q"))),
                    trait=str(entry.get("trait", "E")),
                    text=str(entry.get("text", "")),
                    reverse_scored=bool(entry.get("reverse_scored", False)),
                )
            )
        return built


def parse_likert_responses(
    response: str, questionnaire: Sequence[QuestionnaireItem]
) -> Dict[str, float]:
    if not questionnaire:
        return {}
    digits = [int(match[0]) for match in LIKERT_PATTERN.findall(response)]
    trait_scores: Dict[str, List[float]] = {}
    for item, raw in zip(questionnaire, digits):
        score = likert_to_score(raw, reverse=item.reverse_scored)
        trait_scores.setdefault(item.trait, []).append(score)
    averaged: Dict[str, float] = {}
    for trait, values in trait_scores.items():
        averaged[trait] = sum(values) / len(values)
    return averaged


def likert_to_score(value: int, *, reverse: bool = False) -> float:
    clamped = min(5, max(1, value))
    normalized = (clamped - 3) / 2.0
    if reverse:
        normalized *= -1
    return normalized


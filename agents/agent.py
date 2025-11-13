"""Agent loop wiring observation → reflection → planning → action.

Steering happens through ``persona_alphas`` (contrastive activation addition)
rather than by injecting persona descriptions or coefficient dumps into
prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from agents.language_backend import GenerationResult, LanguageBackend
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion, Planner
from agents.retrieval import MemoryRetriever
from safety.governor import SafetyGovernor
from schemas.agent import AgentState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from schemas.objectives import Objective
else:  # pragma: no cover - runtime fallback for optional dependency
    Objective = object
from schemas.logs import SafetyEvent


@dataclass
class ActionDecision:
    action_type: str
    params: Dict[str, str]
    utterance: str
    prompt: str
    tokens_in: int
    tokens_out: int
    steering_snapshot: Dict[str, float]
    layers_used: List[int]
    safety_event: Optional[SafetyEvent]


class Agent:
    def __init__(
        self,
        run_id: str,
        state: AgentState,
        language_backend: LanguageBackend,
        memory: MemoryStore,
        retriever: MemoryRetriever,
        planner: Planner,
        safety_governor: SafetyGovernor,
        max_new_tokens: int = 120,
        reflect_every_n_ticks: int = 1,
    ):
        self.run_id = run_id
        self.state = state
        self.language_backend = language_backend
        self.memory = memory
        self.retriever = retriever
        self.planner = planner
        self.safety_governor = safety_governor
        self.max_new_tokens = max_new_tokens
        self.reflect_every_n_ticks = reflect_every_n_ticks
        self._last_plan_suggestion: Optional[PlanSuggestion] = None

    # ---- persona helpers ----

    def persona_alphas(self) -> Dict[str, float]:
        base = self.state.persona_coeffs.model_dump()
        alphas = {}
        for trait, coef in base.items():
            delta = self.state.active_alpha_overrides.get(trait, 0.0)
            alphas[trait] = self._clamp(coef + delta)
        return alphas

    def _clamp(self, value: float) -> float:
        clip = self.safety_governor.config.alpha_clip
        return max(-clip, min(clip, value))

    def apply_alpha_delta(self, delta: Dict[str, float]) -> None:
        for trait, change in delta.items():
            self.state.active_alpha_overrides[trait] = self._clamp(
                self.state.active_alpha_overrides.get(trait, 0.0) + change
            )

    # ---- cognitive loop ----

    def perceive(self, observation: str, tick: int) -> None:
        importance = min(1.0, 0.3 + 0.05 * len(observation.split()))
        self.memory.add_event(self.state.agent_id, "observation", tick, observation, importance)

    def reflect_and_plan(
        self,
        tick: int,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
    ) -> PlanSuggestion:
        # Skip reflection if not on reflection cycle and we have a cached plan
        should_reflect = (tick % self.reflect_every_n_ticks == 0)

        if not should_reflect and self._last_plan_suggestion is not None:
            # Reuse last plan suggestion (fast path)
            return self._last_plan_suggestion

        # Full reflection path
        summary, events = self.retriever.summarize(self.state.goals, current_tick=tick)
        reflection_text = f"Focus: {', '.join(self.state.goals) or 'open exploration'}"
        implications = [f"Reference memory {ev.memory_id}" for ev in events]
        self.memory.add_reflection(self.state.agent_id, tick, reflection_text, implications=implications)
        suggestion = self.planner.plan(
            self.state.goals,
            summary,
            current_location=current_location,
            active_objective=active_objective,
            tick=tick,
        )
        self.memory.add_plan(self.state.agent_id, tick, tick + 3, [suggestion.action_type])
        self._last_plan_suggestion = suggestion
        return suggestion

    def _build_prompt(
        self,
        observation: str,
        suggestion: PlanSuggestion,
        *,
        current_location: Optional[str] = None,
    ) -> str:
        goals_text = ", ".join(self.state.goals) or "explore town"
        location_text = current_location or "unknown"
        return (
            "System: Write a concise, natural first-person message (no 'You:' prefix, no hashtags).\n"
            "- Keep perspective consistent and avoid narrating stage directions.\n"
            "- Do not contradict the stated current location.\n"
            f"Current location: {location_text}\n"
            f"Current goals: {goals_text}\n"
            f"Observation: {observation}\n"
            f"Intended utterance guidance: {suggestion.utterance}\n"
        )

    def generate(self, prompt: str, alphas: Dict[str, float]) -> GenerationResult:
        return self.language_backend.generate(prompt, self.max_new_tokens, alphas)

    def act(
        self,
        observation: str,
        tick: int,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
    ) -> ActionDecision:
        self.perceive(observation, tick)
        suggestion = self.reflect_and_plan(
            tick,
            current_location=current_location,
            active_objective=active_objective,
        )
        # Add a brief pre-talk sync at the very beginning to reduce immediate moves.
        if tick == 0 and suggestion.action_type == "move":
            suggestion = PlanSuggestion(
                action_type="talk",
                params={"utterance": "Quick sync: confirm roles and plan, then head out."},
                utterance="Let's quickly align on roles and objectives before moving.",
            )

        prompt = self._build_prompt(
            observation,
            suggestion,
            current_location=current_location,
        )
        alphas = self.persona_alphas()
        generation = self.generate(prompt, alphas)
        safety_event = self.safety_governor.evaluate(
            run_id=self.run_id,
            agent_id=self.state.agent_id,
            text=generation.text,
            tick=tick,
            current_alphas=alphas,
        )
        if safety_event:
            self.apply_alpha_delta(safety_event.applied_alpha_delta)
        params = dict(suggestion.params)
        if suggestion.action_type == "talk":
            params["utterance"] = generation.text
        return ActionDecision(
            action_type=suggestion.action_type,
            params=params,
            utterance=generation.text,
            prompt=prompt,
            tokens_in=generation.tokens_in,
            tokens_out=generation.tokens_out,
            steering_snapshot=alphas,
            layers_used=self.language_backend.layers_used(),
            safety_event=safety_event,
        )

"""Agent loop wiring observation → reflection → planning → action.

Steering happens through ``persona_alphas`` (contrastive activation addition)
rather than by injecting persona descriptions or coefficient dumps into
prompts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Sequence, Tuple

from agents.language_backend import GenerationResult, LanguageBackend
from utils.sanitize import sanitize_agent_output
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion, Planner
from agents.retrieval import MemoryRetriever
from safety.governor import SafetyGovernor
from schemas.agent import AgentState
from env.world import RoomUtterance

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
    prompt_text: str
    prompt_hash: str
    tokens_in: int
    tokens_out: int
    steering_snapshot: Dict[str, float]
    layers_used: List[int]
    safety_event: Optional[SafetyEvent]
    plan_metadata: Dict[str, object]
    reflection_summary: Optional[str]
    reflection_implications: List[str]
    probe_id: Optional[str] = None
    probe_kind: Optional[str] = None


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
        self._last_reflection: Optional[Tuple[str, List[str]]] = None
        self._last_plan_location: Optional[str] = None

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
        rule_context: Optional[List[str]] = None,
    ) -> PlanSuggestion:
        # Skip reflection if not on reflection cycle and we have a cached plan
        should_reflect = (tick % self.reflect_every_n_ticks == 0)

        if (
            not should_reflect
            and self._last_plan_suggestion is not None
            and self._last_plan_location == (current_location or "")
        ):
            # Reuse last plan suggestion (fast path)
            return self._last_plan_suggestion

        # Full reflection path
        summary, events = self.retriever.summarize(
            self.state.goals,
            current_tick=tick,
            focus_terms=[current_location] if current_location else None,
        )
        reflection_text = f"Focus: {', '.join(self.state.goals) or 'open exploration'}"
        implications = [f"Reference memory {ev.memory_id}" for ev in events]
        self.memory.add_reflection(self.state.agent_id, tick, reflection_text, implications=implications)
        self._last_reflection = (summary, implications)
        suggestion = self.planner.plan(
            self.state.goals,
            summary,
            current_location=current_location,
            active_objective=active_objective,
            tick=tick,
            rule_context=rule_context,
        )
        self.memory.add_plan(self.state.agent_id, tick, tick + 3, [suggestion.action_type])
        self._last_plan_suggestion = suggestion
        self._last_plan_location = current_location or ""
        return suggestion

    def _build_prompt(
        self,
        observation: str,
        suggestion: PlanSuggestion,
        *,
        current_location: Optional[str] = None,
        recent_dialogue: Optional[Sequence[RoomUtterance]] = None,
    ) -> str:
        goals_text = ", ".join(self.state.goals) or "explore town"
        location_text = current_location or "unknown"
        agent_name = self.state.agent_id
        dialogue_section = ""
        if recent_dialogue:
            formatted_dialogue = "\n".join(
                f"- {entry.speaker} (tick {entry.tick}): {entry.content}"
                for entry in recent_dialogue
            )
            dialogue_section = f"Recent dialogue:\n{formatted_dialogue}\n"
        param_text = ", ".join(f"{k}={v}" for k, v in suggestion.params.items()) or "none"
        action_directives = [
            "NEXT ACTION DIRECTIVE:",
            f"- Action type: {suggestion.action_type}",
            f"- Parameters: {param_text}",
            f"- Utterance guidance: {suggestion.utterance}",
        ]
        if suggestion.action_type == "trade":
            action_directives.append("- Craft a <=100 character trade offer that states item, qty, and price.")
            action_directives.append("- No narration or extra dialogue—just the trade text you would post.")
        elif suggestion.action_type != "talk":
            action_directives.append("- Keep it to one concise sentence describing what you do next.")
        action_section = "\n".join(action_directives)
        return (
            f"System: You are {agent_name}. Reply with ONLY your own single response as {agent_name}.\n"
            "IMPORTANT RULES:\n"
            f"- Write ONLY what {agent_name} would say; never script other agents\n"
            "- No prefixes like 'You:' or stage directions\n"
            "- Never write multi-turn conversations or imagine replies\n"
            "- Always speak in first-person ('I...') and stay in character\n"
            "- Keep responses concise and consistent with your location\n"
            f"\n{action_section}\n"
            f"\nCurrent location: {location_text}\n"
            f"Current goals: {goals_text}\n"
            f"Observation: {observation}\n"
            f"{dialogue_section}"
            f"{agent_name}'s response:"
        )

    def generate(self, prompt: str, alphas: Dict[str, float]) -> GenerationResult:
        return self.language_backend.generate(prompt, self.max_new_tokens, alphas)

    def act(
        self,
        observation: str,
        tick: int,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
        recent_dialogue: Optional[Sequence[RoomUtterance]] = None,
        rule_context: Optional[List[str]] = None,
    ) -> ActionDecision:
        self.perceive(observation, tick)
        suggestion = self.reflect_and_plan(
            tick,
            current_location=current_location,
            active_objective=active_objective,
            rule_context=rule_context,
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
            recent_dialogue=recent_dialogue,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        alphas = self.persona_alphas()
        generation = self.generate(prompt, alphas)
        cleaned_text = sanitize_agent_output(generation.text)
        # Fallback if cleaning removed everything
        if not cleaned_text:
            cleaned_text = suggestion.utterance or ""
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
            params["utterance"] = cleaned_text
        plan_metadata = suggestion.to_metadata()
        cached_reflection = self._last_reflection or ("", [])
        return ActionDecision(
            action_type=suggestion.action_type,
            params=params,
            utterance=cleaned_text,
            prompt_text=prompt,
            prompt_hash=prompt_hash,
            tokens_in=generation.tokens_in,
            tokens_out=generation.tokens_out,
            steering_snapshot=alphas,
            layers_used=self.language_backend.layers_used(),
            safety_event=safety_event,
            plan_metadata=plan_metadata,
            reflection_summary=cached_reflection[0],
            reflection_implications=list(cached_reflection[1]),
        )

"""Agent loop wiring observation → reflection → planning → action.

Steering happens through ``persona_alphas`` (contrastive activation addition)
rather than by injecting persona descriptions or coefficient dumps into
prompts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Sequence, Tuple

from agents.language_backend import GenerationResult, LanguageBackend
from utils.sanitize import sanitize_agent_output
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion, Planner
from agents.retrieval import MemoryRetriever
from safety.governor import SafetyGovernor
from schemas.agent import AgentState, Rule
from env.world import RoomUtterance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from schemas.objectives import Objective
    from orchestrator.meta_manager import AlignmentContext
else:  # pragma: no cover - runtime fallback for optional dependency
    Objective = object
    AlignmentContext = object
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
        suppress_alphas: bool = False,
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
        self._suppress_alphas = suppress_alphas
        self._last_plan_suggestion: Optional[PlanSuggestion] = None
        self._last_reflection: Optional[Tuple[str, List[str]]] = None
        self._last_plan_location: Optional[str] = None
        self._last_plan_tick: Optional[int] = None
        self._last_observation: Optional[str] = None
        self._last_reflection_tick: Optional[int] = None
        self._last_alignment_tick: Optional[int] = None
        self._initial_sync_completed: bool = False
        self._last_sync_tick: Optional[int] = None

    # ---- persona helpers ----

    def persona_alphas(self) -> Dict[str, float]:
        if self._suppress_alphas:
            return {trait: 0.0 for trait in self.state.persona_coeffs.model_dump().keys()}
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
        rule_context: Optional[List[Rule]] = None,
        observation: Optional[str] = None,
        recent_dialogue: Optional[Sequence[RoomUtterance]] = None,
    ) -> PlanSuggestion:
        # Skip reflection if not on reflection cycle and we have a cached plan
        should_reflect = (tick % self.reflect_every_n_ticks == 0)

        if (
            not should_reflect
            and self._last_plan_suggestion is not None
            and self._last_plan_location == (current_location or "")
            and self._last_plan_tick is not None
            and tick - self._last_plan_tick <= 1
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
        self._last_reflection_tick = tick
        observation_keywords = self._extract_observation_keywords(observation, recent_dialogue)
        suggestion = self.planner.plan(
            self.state.goals,
            summary,
            current_location=current_location,
            active_objective=active_objective,
            tick=tick,
            rule_context=rule_context,
            last_reflection_tick=self._last_reflection_tick,
            last_alignment_tick=self._last_alignment_tick,
            observation_keywords=observation_keywords if observation_keywords else None,
            agent_id=self.state.agent_id,
        )
        self.memory.add_plan(self.state.agent_id, tick, tick + 3, [suggestion.action_type])
        self._last_plan_suggestion = suggestion
        self._last_plan_location = current_location or ""
        self._last_plan_tick = tick
        if suggestion.alignment:
            self._last_alignment_tick = tick
        if observation is not None:
            self._last_observation = observation
        return suggestion

    def _extract_observation_keywords(
        self,
        observation: Optional[str],
        recent_dialogue: Optional[Sequence[RoomUtterance]],
        limit: int = 6,
    ) -> List[str]:
        raw_tokens: List[str] = []
        if observation:
            raw_tokens.extend(observation.split())
        if recent_dialogue:
            for entry in recent_dialogue:
                raw_tokens.extend(entry.content.split())
        keywords: List[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            cleaned = "".join(ch for ch in token.lower() if ch.isalpha() or ch == "_")
            if len(cleaned) < 3 or cleaned in seen:
                continue
            keywords.append(cleaned)
            seen.add(cleaned)
            if len(keywords) >= limit:
                break
        return keywords

    def _extract_observation_highlights(
        self,
        observation: Optional[str],
        recent_dialogue: Optional[Sequence[RoomUtterance]],
        *,
        min_items: int = 2,
        max_items: int = 3,
    ) -> List[str]:
        """Return short phrases that anchor the prompt to fresh context."""

        texts: List[str] = []
        if observation:
            texts.append(observation)
        if recent_dialogue:
            texts.extend(entry.content for entry in recent_dialogue[-2:])
        if not texts:
            return []

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "with",
            "for",
            "of",
            "in",
            "on",
            "at",
            "to",
            "from",
            "this",
            "that",
            "these",
            "those",
            "nearby",
            "agents",
            "agent",
            "location",
            "capacity",
            "resources",
            "neighbors",
            "current",
            "recent",
            "activity",
            "room",
            "there",
            "here",
            "its",
            "it's",
            "be",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "tick",
            "agent's",
            "your",
        }

        tokens: List[str] = []
        for text in texts:
            tokens.extend(re.findall(r"[A-Za-z][A-Za-z'-]*", text))

        highlights: List[str] = []
        seen: set[str] = set()
        idx = 0
        while idx < len(tokens) and len(highlights) < max_items:
            token = tokens[idx].lower()
            idx += 1
            if token in stopwords or len(token) < 3:
                continue
            phrase_words = [token]
            lookahead = 0
            while idx < len(tokens) and lookahead < 2:
                next_token = tokens[idx].lower()
                if next_token in stopwords or len(next_token) < 3:
                    idx += 1
                    continue
                phrase_words.append(next_token)
                idx += 1
                lookahead += 1
            phrase = " ".join(phrase_words)
            normalized = phrase.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            highlights.append(phrase.capitalize())

        if len(highlights) < min_items:
            for text in texts:
                clauses = re.split(r"[.!?]", text)
                for clause in clauses:
                    snippet_words = [word for word in clause.split() if word.strip()]
                    if not snippet_words:
                        continue
                    snippet = " ".join(snippet_words[:10]).strip(" .,;:")
                    if len(snippet) < 4:
                        continue
                    normalized = snippet.lower()
                    if normalized in seen:
                        continue
                    highlights.append(snippet)
                    seen.add(normalized)
                    if len(highlights) >= min_items:
                        break
                if len(highlights) >= min_items:
                    break

        return highlights[:max_items]

    def _uses_quick_sync_template(self, suggestion: PlanSuggestion) -> bool:
        utterance_param = suggestion.params.get("utterance") if suggestion.params else None
        if not utterance_param:
            return False
        return utterance_param.strip().lower().startswith("quick sync")

    def _build_prompt(
        self,
        observation: str,
        suggestion: PlanSuggestion,
        *,
        current_location: Optional[str] = None,
        recent_dialogue: Optional[Sequence[RoomUtterance]] = None,
        alignment_context: Optional["AlignmentContext"] = None,
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
        highlights = self._extract_observation_highlights(observation, recent_dialogue)
        highlight_section = ""
        if highlights:
            highlight_lines = ["Key observation highlights:"] + [f"- {item}" for item in highlights]
            highlight_section = "\n".join(highlight_lines) + "\n\n"
        alignment_section = ""
        if alignment_context:
            alignment_lines = ["Alignment guidance:"]
            if alignment_context.global_goals:
                shared_goals = "; ".join(alignment_context.global_goals)
                alignment_lines.append(f"- Shared goals: {shared_goals}")
            if alignment_context.agent_priority:
                alignment_lines.append(
                    f"- Priority for you: {alignment_context.agent_priority}"
                )
            if alignment_context.task_hint:
                alignment_lines.append(f"- Task focus: {alignment_context.task_hint}")
            for hint in alignment_context.planning_hints or []:
                alignment_lines.append(f"- Planning hint: {hint}")
            for reminder in alignment_context.reminders or []:
                alignment_lines.append(f"- {reminder}")
            alignment_section = "\n".join(alignment_lines) + "\n\n"
        param_text = ", ".join(f"{k}={v}" for k, v in suggestion.params.items()) or "none"
        action_directives = [
            "NEXT ACTION DIRECTIVE:",
            f"- Action type: {suggestion.action_type}",
            f"- Parameters: {param_text}",
            f"- Utterance guidance: {suggestion.utterance}",
        ]
        if suggestion.action_type != "talk":
            action_directives.append("- Keep it to one concise sentence describing what you do next.")
        action_section = "\n".join(action_directives)
        penalty_text = (
            "Penalty: Do NOT repeat 'Quick sync…' unless another agent explicitly asked this tick\n"
            if self._uses_quick_sync_template(suggestion)
            else ""
        )
        return (
            f"System: You are {agent_name}. Reply with ONLY your own single response as {agent_name}.\n"
            "IMPORTANT RULES:\n"
            f"- Write ONLY what {agent_name} would say; never script other agents\n"
            "- No prefixes like 'You:' or stage directions\n"
            "- Never write multi-turn conversations or imagine replies\n"
            "- Always speak in first-person ('I...') and stay in character\n"
            "- Keep responses concise and consistent with your location\n"
            + penalty_text
            + alignment_section
            + ("\n" + highlight_section if highlight_section else "\n")
            + f"{action_section}\n"
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
        rule_context: Optional[List[Rule]] = None,
        peers_present: bool = False,
        alignment_context: Optional["AlignmentContext"] = None,
    ) -> ActionDecision:
        self.perceive(observation, tick)
        suggestion = self.reflect_and_plan(
            tick,
            current_location=current_location,
            active_objective=active_objective,
            rule_context=rule_context,
            observation=observation,
            recent_dialogue=recent_dialogue,
        )
        # Add a brief pre-talk sync at the very beginning to reduce immediate moves.
        location_hint = (current_location or self.state.location_id or "town_square").replace("_", " ")
        if (
            tick == 0
            and suggestion.action_type == "move"
            and peers_present
            and not self._initial_sync_completed
            and self._last_sync_tick != tick
        ):
            quick_sync_text = (
                f"Quick sync ({self.state.agent_id} @ {location_hint}): confirm roles and plan before moving."
            )
            suggestion = PlanSuggestion(
                action_type="talk",
                params={
                    "utterance": quick_sync_text,
                    "topic": f"initial_sync:{self.state.agent_id}:{location_hint}",
                },
                utterance=f"Let's quickly align here in the {location_hint} before moving.",
            )
            self._initial_sync_completed = True
            self._last_sync_tick = tick

        prompt = self._build_prompt(
            observation,
            suggestion,
            current_location=current_location,
            recent_dialogue=recent_dialogue,
            alignment_context=alignment_context,
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

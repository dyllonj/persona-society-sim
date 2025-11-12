"""Agent loop wiring observation → reflection → planning → action."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from agents.memory import MemoryStore
from agents.planner import PlanSuggestion, Planner
from schemas.agent import AgentState


class Agent:
    def __init__(
        self,
        state: AgentState,
        model,
        tokenizer,
        memory: MemoryStore,
        planner: Planner,
        steering_controller: Optional[Any] = None,
    ):
        self.state = state
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.planner = planner
        self.steering_controller = steering_controller

    def perceive(self, observation: str, tick: int) -> None:
        importance = min(1.0, 0.3 + 0.05 * len(observation.split()))
        self.memory.add_event(self.state.agent_id, "observation", tick, observation, importance)

    def reflect_and_plan(self, tick: int) -> PlanSuggestion:
        recent = self.memory.recent_events(limit=5)
        summary = " \n".join(ev.text for ev in recent)
        reflection_text = f"Key focus areas: {', '.join(self.state.goals) or 'open exploration'}"
        self.memory.add_reflection(self.state.agent_id, tick, reflection_text, implications=[reflection_text])
        suggestion = self.planner.plan(self.state.goals, summary)
        self.memory.add_plan(self.state.agent_id, tick, tick + 3, [suggestion.action_type])
        return suggestion

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def act(self, observation: str, tick: int) -> Dict[str, Any]:
        self.perceive(observation, tick)
        suggestion = self.reflect_and_plan(tick)
        prompt = (
            f"System: Maintain persona per coefficients {self.state.persona_coeffs.model_dump()}\n"
            f"Agent: {self.state.display_name}\n"
            f"Context: {observation}\n"
            f"Intent: {suggestion.utterance}\n"
        )
        utterance = self.generate(prompt)
        return {
            "action_type": suggestion.action_type,
            "params": suggestion.params,
            "utterance": utterance,
        }

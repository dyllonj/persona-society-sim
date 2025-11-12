"""Simple planner translating reflections/goals into next actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PlanSuggestion:
    action_type: str
    params: Dict[str, str]
    utterance: str


class Planner:
    def __init__(self, default_location: str = "town_square"):
        self.default_location = default_location

    def plan(self, goals: List[str], memory_summary: str) -> PlanSuggestion:
        if goals:
            goal = goals[0]
            utterance = f"I am focusing on {goal.lower()} right now."
        else:
            goal = "socialize"
            utterance = "I'm taking a moment to chat with whoever is nearby."

        if "meet" in goal.lower():
            action = "move"
            params = {"destination": "community_center"}
        elif "project" in goal.lower() or "task" in goal.lower():
            action = "work"
            params = {"task": goal}
        elif "market" in memory_summary.lower():
            action = "trade"
            params = {"item": "produce", "qty": "1"}
        else:
            action = "talk"
            params = {"topic": goal}
        return PlanSuggestion(action_type=action, params=params, utterance=utterance)

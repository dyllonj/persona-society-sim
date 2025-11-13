"""Simple planner translating reflections/goals into next actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from schemas.objectives import Objective
else:  # pragma: no cover - runtime fallback when schemas are unavailable
    Objective = Any


@dataclass
class PlanSuggestion:
    action_type: str
    params: Dict[str, str]
    utterance: str


class Planner:
    def __init__(self, default_location: str = "town_square"):
        self.default_location = default_location
        self._objective_heuristics = {
            "collaborate": {
                "location": "community_center",
                "action": "talk",
                "utterance": "Let's collaborate on ways to strengthen our community.",
            },
            "gather": {
                "location": "market",
                "action": "trade",
                "item": "supplies",
                "qty": "1",
                "utterance": "I'm arranging trades to gather the supplies we need.",
            },
            "research": {
                "location": "library",
                "action": "talk",
                "utterance": "I'm focusing on research findings to share with others.",
            },
            "explore": {
                "location": "town_square",
                "action": "move",
                "utterance": "I'm exploring different areas to learn more about the town.",
            },
            "socialize": {
                "location": "community_center",
                "action": "talk",
                "utterance": "I'm here to meet and connect with others.",
            },
            "trade": {
                "location": "market",
                "action": "trade",
                "item": "goods",
                "qty": "1",
                "utterance": "I'm looking to trade and support the local market.",
            },
            "work": {
                "location": "community_center",
                "action": "work",
                "task": "community project",
                "utterance": "I'm contributing my time to help with town projects.",
            },
            "community": {
                "location": "community_center",
                "action": "talk",
                "utterance": "I'm working to build stronger community connections.",
            },
        }

    def plan(
        self,
        goals: List[str],
        memory_summary: str,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
    ) -> PlanSuggestion:
        location = current_location or self.default_location

        if active_objective:
            objective_plan = self._plan_for_objective(active_objective, location)
            if objective_plan:
                return objective_plan

        if goals:
            goal = goals[0]
            utterance = f"I am focusing on {goal.lower()} right now."
        else:
            goal = "socialize"
            utterance = "I'm taking a moment to chat with whoever is nearby."

        if "meet" in goal.lower() and location != "community_center":
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
            params = {"utterance": goal}
        return PlanSuggestion(action_type=action, params=params, utterance=utterance)

    def _plan_for_objective(
        self, objective: Objective, current_location: str
    ) -> Optional[PlanSuggestion]:
        heuristic = self._objective_heuristics.get(objective.type.lower())
        if not heuristic:
            return None

        target_location = heuristic["location"]
        description = objective.description
        utterance = heuristic.get(
            "utterance",
            f"I'm progressing on {objective.type.lower()} by following the plan.",
        )

        if current_location != target_location:
            move_params = {"destination": target_location}
            move_utterance = (
                f"Heading to the {target_location.replace('_', ' ')} to work on {description.lower()}."
            )
            return PlanSuggestion("move", move_params, move_utterance)

        action_type = heuristic["action"]
        if action_type == "talk":
            params = {"utterance": utterance}
        elif action_type == "trade":
            params = {
                "item": heuristic.get("item", "goods"),
                "qty": heuristic.get("qty", "1"),
            }
        elif action_type == "work":
            params = {"task": heuristic.get("task", "project")}
        elif action_type == "move":
            # For explore objectives, pick a destination different from current location
            destinations = ["town_square", "community_center", "market", "library"]
            available = [d for d in destinations if d != current_location]
            params = {"destination": available[0] if available else "town_square"}
        else:
            # Fallback for unknown action types
            params = {}

        return PlanSuggestion(action_type=action_type, params=params, utterance=utterance)

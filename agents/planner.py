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
                "action": "research",
                "utterance": "I'm focusing on research findings to share with others.",
            },
            "research_facts": {
                "location": "library",
                "action": "research",
                "utterance": "I'll look up documents relevant to our fact targets.",
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
            "policy": {
                "location": "community_center",
                "action": "fill_field",
                "utterance": "Completing the compliance checklist.",
            },
            "navigation": {
                "location": "town_square",
                "action": "scan",
                "utterance": "Covering new ground to find scan tokens.",
            },
        }
        self._nav_cycle = ["town_square", "market", "library", "community_center"]

    def plan(
        self,
        goals: List[str],
        memory_summary: str,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
        tick: int = 0,
    ) -> PlanSuggestion:
        location = current_location or self.default_location

        if active_objective:
            objective_plan = self._plan_for_objective(active_objective, location, tick)
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
        self, objective: Objective, current_location: str, tick: int
    ) -> Optional[PlanSuggestion]:
        obj_type = objective.type.lower()
        if obj_type == "policy":
            return self._plan_policy_objective(objective, current_location)
        if obj_type == "navigation":
            return self._plan_navigation_objective(objective, current_location, tick)

        heuristic = self._objective_heuristics.get(obj_type)
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
        # Special handling for research-style objectives: schedule research → cite → submit
        if obj_type in {"research", "research_facts"}:
            step = tick % 4
            if step in (0, 1):
                return PlanSuggestion("research", {"query": description.split()[0] if description else ""}, utterance)
            if step == 2:
                return PlanSuggestion("cite", {}, "I'll add a supporting citation.")
            return PlanSuggestion("submit_report", {}, "Submitting a brief report of findings.")

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

    def _plan_policy_objective(
        self, objective: Objective, current_location: str
    ) -> Optional[PlanSuggestion]:
        target_location = "community_center"
        if current_location != target_location:
            return PlanSuggestion(
                "move",
                {"destination": target_location},
                "Heading to the community center to finish the checklist.",
            )
        fields_required = objective.requirements.get("fill_field", 0)
        fields_completed = objective.progress.get("fill_field", 0)
        if fields_completed < fields_required:
            field_name = f"policy_field_{fields_completed + 1}"
            params = {
                "field_name": field_name,
                "value": f"Action plan item {fields_completed + 1}",
            }
            utterance = f"Filling checklist field {field_name}."
            return PlanSuggestion("fill_field", params, utterance)
        if objective.requirements.get("propose_plan"):
            needed = objective.requirements["propose_plan"]
            done = objective.progress.get("propose_plan", 0)
            if done < needed:
                return PlanSuggestion(
                    "propose_plan",
                    {"summary": objective.description},
                    "Drafting the compliance plan summary.",
                )
        if objective.requirements.get("submit_plan"):
            submitted = objective.progress.get("submit_plan", 0)
            if submitted < objective.requirements["submit_plan"]:
                return PlanSuggestion(
                    "submit_plan",
                    {},
                    "Submitting the checklist for approval.",
                )
        return None

    def _plan_navigation_objective(
        self, objective: Objective, current_location: str, tick: int
    ) -> Optional[PlanSuggestion]:
        scan_goal = objective.requirements.get("scan", 0)
        scans_completed = objective.progress.get("scan", 0)
        # Rotate through known destinations based on tick + progress to avoid crowding
        target_index = (scans_completed + tick) % len(self._nav_cycle)
        target_location = self._nav_cycle[target_index]
        if current_location != target_location and tick % 2 == 0:
            utterance = f"Moving to {target_location.replace('_', ' ')} to scan for tokens."
            return PlanSuggestion("move", {"destination": target_location}, utterance)
        if scans_completed < scan_goal:
            return PlanSuggestion("scan", {}, "Scanning the area for discovery tokens.")
        # Once scans are complete, keep exploring to assist others
        next_location = self._nav_cycle[(target_index + 1) % len(self._nav_cycle)]
        utterance = "Coverage complete; relocating to coordinate with others."
        return PlanSuggestion("move", {"destination": next_location}, utterance)

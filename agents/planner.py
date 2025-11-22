"""Simple planner translating reflections/goals into next actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from schemas.objectives import Objective
    from schemas.agent import Rule
else:  # pragma: no cover - runtime fallback when schemas are unavailable
    Objective = Any
    Rule = Any


@dataclass
class PlanSuggestion:
    action_type: str
    params: Dict[str, str]
    utterance: str
    alignment: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the plan suggestion."""
        return {
            "action_type": self.action_type,
            "params": dict(self.params),
            "utterance": self.utterance,
            "alignment": self.alignment,
        }


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
                "location": "library",
                "action": "cite",
                "utterance": "I'm gathering references in the library to cite for our report.",
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
            "report": {
                "location": "library",
                "action": "submit_report",
                "utterance": "I'm compiling our findings into a report at the library.",
            },
        }
        self._role_templates = {
            "research": {
                "move": "Heading to the library to keep our research momentum.",
                "research": "Investigating sources that support our research goals.",
                "cite": "Recording a citation to back up our findings.",
                "submit_report": "Submitting a concise research report for the team.",
            },
            "policy": {
                "move": "Relocating to the community center to own the compliance checklist.",
                "fill_field": "Capturing the next compliance detail in the checklist.",
                "propose_plan": "Drafting the policy summary before submission.",
                "submit_plan": "Submitting the compliance plan for approval.",
            },
            "navigation": {
                "move": "Rotating to {destination} to extend our scan coverage.",
                "scan": "Scanning this room to exhaust the remaining tokens.",
            },
        }
        known_locations = {self.default_location}
        for heuristic in self._objective_heuristics.values():
            target = heuristic.get("location")
            if target:
                known_locations.add(target)
        nav_defaults = ["town_square", "community_center", "library"]
        cycle = [room for room in nav_defaults if room in known_locations]
        if self.default_location in cycle:
            cycle.remove(self.default_location)
            cycle.insert(0, self.default_location)
        elif cycle:
            cycle.insert(0, self.default_location)
        else:
            cycle = [self.default_location]
        self._nav_cycle = cycle

    def plan(
        self,
        goals: List[str],
        memory_summary: str,
        current_location: Optional[str] = None,
        active_objective: Optional[Objective] = None,
        tick: int = 0,
        rule_context: Optional[List[Rule]] = None,
        last_reflection_tick: Optional[int] = None,
        last_alignment_tick: Optional[int] = None,
        observation_keywords: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        planning_hints: Optional[List[str]] = None,
        agent_role: Optional[str] = None,
    ) -> PlanSuggestion:
        location = current_location or self.default_location

        # 0. Forced collaboration override
        if planning_hints and "force_collaboration" in planning_hints:
            return PlanSuggestion(
                action_type="talk",
                params={
                    "utterance": "Let's sync up on our tasks.",
                    "topic": "forced_collab",
                },
                utterance="I need to coordinate with everyone here.",
                alignment=True,
            )

        objective_type: Optional[str] = None
        objective_pending = False
        if active_objective:
            objective_type = getattr(active_objective, "type", None)
            if isinstance(objective_type, str):
                objective_type = objective_type.lower()
            objective_pending = self._objective_pending(active_objective)
            objective_plan = self._plan_for_objective(active_objective, location, tick)
            if objective_plan:
                return self._enforce_role_alignment(
                    objective_plan, agent_role, location, tick
                )

        if rule_context:
            rule_plan = self._plan_from_rules(
                rule_context,
                location,
                objective_pending=objective_pending,
                objective_type=objective_type,
            )
            if rule_plan:
                return self._enforce_role_alignment(
                    rule_plan, agent_role, location, tick
                )

        lowered_summary = memory_summary.lower()
        observation_terms = [hint.lower() for hint in (observation_keywords or [])]

        alignment_blocked = (
            last_reflection_tick is not None
            and last_alignment_tick == last_reflection_tick
        )
        if last_reflection_tick is not None and not alignment_blocked:
            return self._alignment_plan(location, agent_id)

        keyword_plan = self._plan_from_keywords(
            observation_terms, location, agent_role, tick
        )
        if keyword_plan:
            return keyword_plan

        role_plan = self._role_bias_plan(agent_role, location, tick)
        if role_plan:
            return role_plan

        def keyword_score(term: str) -> int:
            normalized = term.lower()
            summary_hits = lowered_summary.count(normalized)
            hint_hits = sum(1 for hint in observation_terms if normalized in hint)
            return summary_hits + hint_hits

        if goals:
            idx = tick % len(goals) if tick is not None else 0
            goal = goals[idx]
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
        elif keyword_score("report") > 0:
            action = "submit_report"
            params = {}
        elif keyword_score("cite") > 0 or keyword_score("reference") > 0:
            action = "cite"
            params = {}
        elif keyword_score("work") > 0 or keyword_score("project") > 0:
            action = "work"
            params = {"task": goal or "community project"}
        elif keyword_score("research") > 0 or keyword_score("library") > 0:
            action = "research"
            params = {"query": (goal.split()[0] if goal else "topic")}
        else:
            action = "talk"
            params = {"utterance": goal}
        return self._enforce_role_alignment(
            PlanSuggestion(action_type=action, params=params, utterance=utterance),
            agent_role,
            location,
            tick,
        )

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
        return self._build_plan_from_heuristic(
            heuristic,
            current_location,
            utterance_override=utterance,
        )

    def _alignment_plan(self, location: str, agent_id: Optional[str]) -> PlanSuggestion:
        friendly_location = location.replace("_", " ")
        speaker = agent_id or "our group"
        utterance = (
            f"Let's align on our plan while {speaker} is at the {friendly_location}."
        )
        params = {
            "utterance": (
                f"Quick sync ({speaker} @ {friendly_location}): confirm goals before next steps."
            ),
            "topic": f"alignment:{friendly_location}:{speaker}",
        }
        return PlanSuggestion("talk", params, utterance, alignment=True)

    def _plan_from_keywords(
        self,
        observation_terms: List[str],
        location: str,
        agent_role: Optional[str],
        tick: int,
    ) -> Optional[PlanSuggestion]:
        if not observation_terms:
            return None

        def has_term(term: str) -> bool:
            return any(term in keyword for keyword in observation_terms)

        if has_term("library") or has_term("research"):
            if location != "library":
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": "library"},
                    "Heading to the library to dig into the research people flagged.",
                )
                return self._enforce_role_alignment(
                    suggestion, agent_role, location, tick
                )
            suggestion = PlanSuggestion(
                "research",
                {"query": "community topics"},
                "I'll investigate the research leads that just came up.",
            )
            return self._enforce_role_alignment(
                suggestion, agent_role, location, tick
            )

        if has_term("cite") or has_term("citation") or has_term("reference"):
            if location != "library":
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": "library"},
                    "Heading to the library so I can cite the sources people requested.",
                )
                return self._enforce_role_alignment(
                    suggestion, agent_role, location, tick
                )
            suggestion = PlanSuggestion(
                "cite",
                {},
                "I'll cite a source to back up our discussion.",
            )
            return self._enforce_role_alignment(
                suggestion, agent_role, location, tick
            )

        if has_term("report") or has_term("brief"):
            if location != "library":
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": "library"},
                    "Heading to the library to wrap up the report folks asked about.",
                )
                return self._enforce_role_alignment(
                    suggestion, agent_role, location, tick
                )
            suggestion = PlanSuggestion(
                "submit_report",
                {},
                "I'll submit a concise report based on our findings.",
            )
            return self._enforce_role_alignment(
                suggestion, agent_role, location, tick
            )

        if has_term("wellbeing"):
            suggestion = PlanSuggestion(
                "work",
                {"task": "wellbeing support"},
                "I'll cover the wellbeing tasks people just surfaced.",
            )
            return self._enforce_role_alignment(
                suggestion, agent_role, location, tick
            )

        if has_term("community"):
            if location != "community_center":
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": "community_center"},
                    "Heading to the community center to follow up.",
                )
                return self._enforce_role_alignment(
                    suggestion, agent_role, location, tick
                )
            suggestion = PlanSuggestion(
                "work",
                {"task": "community center tasks"},
                "I'll help with the community center tasks people flagged.",
            )
            return self._enforce_role_alignment(
                suggestion, agent_role, location, tick
            )

        return None

    def _role_bias_plan(
        self, agent_role: Optional[str], location: str, tick: int
    ) -> Optional[PlanSuggestion]:
        if not agent_role:
            return None

        role = agent_role.lower()
        if role == "research":
            target_location = "library"
            if location != target_location:
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": target_location},
                    self._role_templates["research"]["move"],
                )
                return self._apply_role_templates(suggestion, role)
            step = (tick or 0) % 3
            if step == 0:
                suggestion = PlanSuggestion(
                    "research",
                    {"query": "priority topic"},
                    self._role_templates["research"]["research"],
                )
            elif step == 1:
                suggestion = PlanSuggestion(
                    "cite", {}, self._role_templates["research"]["cite"]
                )
            else:
                suggestion = PlanSuggestion(
                    "submit_report",
                    {},
                    self._role_templates["research"]["submit_report"],
                )
            return self._apply_role_templates(suggestion, role)

        if role == "policy":
            target_location = "community_center"
            if location != target_location:
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": target_location},
                    self._role_templates["policy"]["move"],
                )
                return self._apply_role_templates(suggestion, role)
            step = (tick or 0) % 3
            if step in (0, 1):
                field_number = (tick or 0) % 4 + 1
                suggestion = PlanSuggestion(
                    "fill_field",
                    {
                        "field_name": f"policy_field_{field_number}",
                        "value": f"Ownership note {field_number}",
                    },
                    self._role_templates["policy"]["fill_field"],
                )
            else:
                suggestion = PlanSuggestion(
                    "submit_plan",
                    {},
                    self._role_templates["policy"]["submit_plan"],
                )
            return self._apply_role_templates(suggestion, role)

        if role == "navigation":
            if not self._nav_cycle:
                return None
            target_index = (tick or 0) % len(self._nav_cycle)
            target_location = self._nav_cycle[target_index]
            if location != target_location:
                suggestion = PlanSuggestion(
                    "move",
                    {"destination": target_location},
                    self._role_templates["navigation"]["move"].format(
                        destination=target_location.replace("_", " ")
                    ),
                )
                return self._apply_role_templates(suggestion, role)
            if (tick or 0) % 3 != 2:
                suggestion = PlanSuggestion(
                    "scan", {}, self._role_templates["navigation"]["scan"]
                )
                return self._apply_role_templates(suggestion, role)
            next_location = self._nav_cycle[(target_index + 1) % len(self._nav_cycle)]
            suggestion = PlanSuggestion(
                "move",
                {"destination": next_location},
                "Coverage complete here; rotating rooms to keep scanning.",
            )
            return self._apply_role_templates(suggestion, role)

        return None

    def _apply_role_templates(
        self, suggestion: PlanSuggestion, role: str
    ) -> PlanSuggestion:
        templates = self._role_templates.get(role)
        if not templates:
            return suggestion

        template = templates.get(suggestion.action_type)
        if template:
            try:
                suggestion.utterance = template.format(
                    destination=suggestion.params.get("destination", "").replace(
                        "_", " "
                    ),
                    field=suggestion.params.get("field_name", ""),
                )
            except Exception:  # pragma: no cover - defensive formatting
                suggestion.utterance = template
        return suggestion

    def _enforce_role_alignment(
        self,
        suggestion: PlanSuggestion,
        agent_role: Optional[str],
        location: str,
        tick: int,
    ) -> PlanSuggestion:
        if not agent_role or not suggestion:
            return suggestion

        role = agent_role.lower()
        allowed_actions = {
            "research": {"move", "research", "cite", "submit_report"},
            "policy": {"move", "fill_field", "submit_plan", "propose_plan"},
            "navigation": {"move", "scan"},
        }
        allowed = allowed_actions.get(role)
        if allowed and suggestion.action_type not in allowed:
            replacement = self._role_bias_plan(agent_role, location, tick)
            if replacement:
                return replacement
        return self._apply_role_templates(suggestion, role)

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

    def _plan_from_rules(
        self,
        rule_context: List[Rule],
        current_location: str,
        *,
        objective_pending: bool = False,
        objective_type: Optional[str] = None,
    ) -> Optional[PlanSuggestion]:
        keyword_map = [
            ("library", "research"),
            ("research", "research"),
            ("cite", "gather"),
            ("reference", "gather"),
            ("report", "report"),
            ("brief", "report"),
            ("work", "work"),
            ("community", "community"),
            ("talk", "community"),
            ("scan", "navigation"),
        ]
        for rule in reversed(rule_context):
            rule_text = getattr(rule, "text", "")
            lowered = rule_text.lower()
            priority = getattr(rule, "priority", "mandatory")
            if (
                objective_pending
                and objective_type in {"research", "research_facts"}
                and priority == "advisory"
            ):
                continue
            for keyword, heuristic_key in keyword_map:
                if keyword in lowered:
                    heuristic = self._objective_heuristics.get(heuristic_key)
                    if not heuristic:
                        continue
                    guidance = f"Complying with rule: {rule_text}"
                    suggestion = self._build_plan_from_heuristic(
                        heuristic,
                        current_location,
                        utterance_override=guidance,
                    )
                    if suggestion:
                        return suggestion
        return None

    def _build_plan_from_heuristic(
        self,
        heuristic: Dict[str, str],
        current_location: str,
        utterance_override: Optional[str] = None,
    ) -> Optional[PlanSuggestion]:
        target_location = heuristic.get("location", self.default_location)
        action_type = heuristic.get("action", "talk")
        if current_location != target_location:
            move_utterance = utterance_override or (
                f"Heading to {target_location.replace('_', ' ')} to continue the plan."
            )
            return PlanSuggestion("move", {"destination": target_location}, move_utterance)
        params = self._params_for_action(action_type, heuristic)
        utterance = utterance_override or heuristic.get(
            "utterance", f"Following through on {action_type}."
        )
        return PlanSuggestion(action_type, params, utterance)

    def _params_for_action(self, action_type: str, heuristic: Dict[str, str]) -> Dict[str, str]:
        if action_type == "talk":
            return {"utterance": heuristic.get("utterance", "Let's sync up.")}
        if action_type == "work":
            return {"task": heuristic.get("task", "project")}
        if action_type == "cite":
            doc_id = heuristic.get("doc_id")
            return {"doc_id": doc_id} if doc_id else {}
        if action_type == "submit_report":
            return {}
        if action_type == "move":
            destinations = ["town_square", "community_center", "library"]
            available = [d for d in destinations if d != heuristic.get("location")]
            return {"destination": available[0] if available else self.default_location}
        return {}

    def _objective_pending(self, objective: Objective) -> bool:
        if not objective:
            return False
        checker = getattr(objective, "is_complete", None)
        if callable(checker):
            try:
                return not checker()
            except Exception:  # pragma: no cover - defensive fallback
                pass
        requirements = getattr(objective, "requirements", {}) or {}
        progress = getattr(objective, "progress", {}) or {}
        if isinstance(requirements, dict) and requirements:
            for key, required in requirements.items():
                try:
                    current = progress.get(key, 0) if isinstance(progress, dict) else 0
                except Exception:  # pragma: no cover - defensive fallback
                    current = 0
                if current < required:
                    return True
            return False
        # If we cannot reason about requirements, assume pending to avoid starving objectives
        return True

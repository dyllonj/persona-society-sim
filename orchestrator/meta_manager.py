"""Meta-level coordinator that nudges agents toward shared goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from schemas.objectives import Objective


@dataclass
class AlignmentContext:
    """Guidance from the meta-orchestrator for a single agent."""

    global_goals: List[str] = field(default_factory=list)
    agent_priority: Optional[str] = None
    reminders: List[str] = field(default_factory=list)
    broadcast: Optional[str] = None
    planning_hints: List[str] = field(default_factory=list)
    task_hint: Optional[str] = None


class MetaOrchestrator:
    """Tracks run-level goals and issues alignment nudges to agents."""

    def __init__(
        self,
        global_goals: Optional[List[str]] = None,
        recurring_reminders: Optional[List[str]] = None,
        agent_directives: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.global_goals = list(global_goals or [])
        self.recurring_reminders = list(recurring_reminders or [])
        self.agent_directives: Dict[str, List[str]] = agent_directives or {}
        self.last_broadcast: Optional[str] = None

    def update_global_goals(self, goals: Iterable[str]) -> None:
        """Replace the shared run-level goals the meta agent is pursuing."""

        self.global_goals = list(goals)

    def set_agent_directives(self, agent_id: str, directives: Iterable[str]) -> None:
        """Persist meta-level planning hints for a specific agent."""

        self.agent_directives[agent_id] = list(directives)

    def reprioritize_objectives(
        self, objective_manager, agent_ids: Iterable[str]
    ) -> Dict[str, Optional[Objective]]:
        """Ensure each agent has an active objective, returning updates."""

        updated: Dict[str, Optional[Objective]] = {}
        if not objective_manager:
            return updated

        for agent_id in agent_ids:
            refreshed = objective_manager.ensure_objective(agent_id)
            if refreshed is None:
                refreshed = objective_manager.current_objective(agent_id)
            updated[agent_id] = refreshed
        return updated

    def broadcast_reminder(self, tick: int) -> Optional[str]:
        """Emit a recurring alignment reminder for the current tick."""

        if not self.recurring_reminders:
            self.last_broadcast = None
            return None
        idx = tick % len(self.recurring_reminders)
        self.last_broadcast = self.recurring_reminders[idx]
        return self.last_broadcast

    def alignment_directives(
        self, tick: int, agents: Dict[str, object], objective_manager=None
    ) -> Dict[str, AlignmentContext]:
        """Produce per-agent alignment contexts for the current tick."""

        broadcast = self.broadcast_reminder(tick)
        active_objectives = self.reprioritize_objectives(
            objective_manager, agents.keys()
        )

        contexts: Dict[str, AlignmentContext] = {}
        for agent_id in agents.keys():
            reminders: List[str] = []
            priority: Optional[str] = None
            planning_hints = list(self.agent_directives.get(agent_id, []))
            task_hint: Optional[str] = None
            objective = active_objectives.get(agent_id) if active_objectives else None
            if objective and isinstance(objective, Objective):
                priority = objective.description
                reminders.append(f"Stay focused on: {objective.description}")
                task_hint = self._task_assignment_hint(objective)
            if broadcast:
                reminders.append(f"Meta reminder: {broadcast}")

            contexts[agent_id] = AlignmentContext(
                global_goals=list(self.global_goals),
                agent_priority=priority,
                reminders=reminders,
                broadcast=broadcast,
                planning_hints=planning_hints,
                task_hint=task_hint,
            )
        return contexts

    def _task_assignment_hint(self, objective: Objective) -> Optional[str]:
        """Summarize the next outstanding requirement to push progress forward."""

        if not objective.requirements:
            return None
        for key, required in objective.requirements.items():
            progress = objective.progress.get(key, 0)
            remaining = max(0, required - progress)
            if remaining:
                return f"Advance '{key}' ({remaining} remaining)"
        return "Wrap up and report progress"


__all__ = ["AlignmentContext", "MetaOrchestrator"]

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
        self.last_room_activity: Dict[str, int] = {}
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
        self,
        tick: int,
        agents: Dict[str, object],
        objective_manager=None,
        world_state: Optional[Dict] = None,
    ) -> Dict[str, AlignmentContext]:
        """Produce per-agent alignment contexts for the current tick."""

        broadcast = self.broadcast_reminder(tick)
        active_objectives = self.reprioritize_objectives(
            objective_manager, agents.keys()
        )

        # Identify rooms that need forced collaboration
        force_collab_rooms = set()
        if world_state:
            # Group agents by room
            room_occupants: Dict[str, List[str]] = {}
            for agent_id, loc_id in world_state.items():
                if loc_id not in room_occupants:
                    room_occupants[loc_id] = []
                room_occupants[loc_id].append(agent_id)
            
            # Check for silence in occupied rooms
            for room_id, occupants in room_occupants.items():
                if len(occupants) > 1:
                    last_active = self.last_room_activity.get(room_id, -1)
                    # If silent for > 3 ticks (configurable?), force collaboration
                    if tick - last_active > 3:
                        force_collab_rooms.add(room_id)

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
            
            # Inject forced collaboration hint if applicable
            current_loc = world_state.get(agent_id) if world_state else None
            if current_loc and current_loc in force_collab_rooms:
                planning_hints.append("force_collaboration")
                reminders.append("Manager: You are too quiet. Coordinate with others now.")

            contexts[agent_id] = AlignmentContext(
                global_goals=list(self.global_goals),
                agent_priority=priority,
                reminders=reminders,
                broadcast=broadcast,
                planning_hints=planning_hints,
                task_hint=task_hint,
            )
        return contexts

    def observe_tick(self, tick: int, logs: List[object]) -> None:
        """Update internal state based on what happened this tick."""
        # logs are ActionLog objects, but we use object to avoid circular imports if possible
        # or just assume they have 'action_type' and 'agent_id'
        
        # We need to know where the action happened. 
        # The runner logs have 'info' which might contain location, or we rely on the runner passing location map?
        # Actually, ActionLog doesn't strictly have location. 
        # But we can infer "talk" actions imply activity in the agent's current room.
        # For now, we'll rely on the runner passing a location map or similar, 
        # OR we just look at the logs if they have enough info.
        # Let's assume we can't easily get location from logs alone without looking up agent state.
        # So we'll just iterate logs and if it's a 'talk', we mark the room as active.
        # Wait, we don't have the room in the log directly easily unless we put it there.
        # The runner puts 'encounter_room' in instrumentation, but maybe not in ActionLog directly?
        # Let's check ActionLog definition in schemas/logs.py if needed, but for now
        # let's assume we can pass a map of {agent_id: room_id} to this method too?
        # Or simpler: The runner calls this, and the runner knows everything.
        pass 

    def update_activity(self, tick: int, room_activity: Iterable[str]) -> None:
        """Record which rooms had social activity this tick."""
        for room_id in room_activity:
            self.last_room_activity[room_id] = tick

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

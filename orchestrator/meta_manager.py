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


@dataclass
class RoleDirectives:
    """Role-specific nudges the meta-orchestrator can apply."""

    planning_hints: List[str] = field(default_factory=list)
    reminders: List[str] = field(default_factory=list)


class MetaOrchestrator:
    """Tracks run-level goals and issues alignment nudges to agents."""

    def __init__(
        self,
        global_goals: Optional[List[str]] = None,
        recurring_reminders: Optional[List[str]] = None,
        agent_directives: Optional[Dict[str, List[str]]] = None,
        role_playbook: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
        environment: Optional[str] = None,
    ) -> None:
        self.global_goals = list(global_goals or [])
        self.recurring_reminders = list(recurring_reminders or [])
        self.agent_directives: Dict[str, List[str]] = agent_directives or {}
        self.last_room_activity: Dict[str, int] = {}
        self.last_broadcast: Optional[str] = None
        base_playbook = self.default_role_playbook()
        if role_playbook:
            base_playbook = self.merge_playbooks(base_playbook, role_playbook)
        self.role_playbook = self._normalize_playbook(base_playbook)
        self.environment = environment

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
        agent_roles: Optional[Dict[str, str]] = None,
        environment: Optional[str] = None,
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

        env_key = (environment or self.environment or "").lower()
        contexts: Dict[str, AlignmentContext] = {}
        for agent_id in agents.keys():
            reminders: List[str] = []
            priority: Optional[str] = None
            planning_hints = list(self.agent_directives.get(agent_id, []))
            task_hint: Optional[str] = None
            role = None
            if agent_roles:
                role = agent_roles.get(agent_id)
            if role is None:
                role = getattr(getattr(agents.get(agent_id), "state", None), "role", None)
            role_directives = self._role_directives(env_key, role)
            if role_directives.reminders:
                reminders.extend(role_directives.reminders)
            if role_directives.planning_hints:
                planning_hints.extend(role_directives.planning_hints)
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

    @staticmethod
    def merge_playbooks(
        base: Dict[str, Dict[str, Dict[str, List[str]]]],
        overrides: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]],
    ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Shallow merge role directives with override precedence."""

        merged: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for env_key, roles in (base or {}).items():
            if not isinstance(roles, dict):
                continue
            merged[env_key] = {
                role: {
                    "planning_hints": list((directives or {}).get("planning_hints", [])),
                    "reminders": list((directives or {}).get("reminders", [])),
                }
                for role, directives in roles.items()
                if isinstance(directives, dict)
            }

        for env_key, roles in (overrides or {}).items():
            if not isinstance(roles, dict):
                continue
            bucket = merged.setdefault(env_key, {})
            for role, directives in roles.items():
                if not isinstance(directives, dict):
                    continue
                baseline = bucket.get(role, {"planning_hints": [], "reminders": []})
                hints = directives.get("planning_hints")
                reminders = directives.get("reminders")
                bucket[role] = {
                    "planning_hints": list(
                        hints if hints is not None else baseline.get("planning_hints", [])
                    ),
                    "reminders": list(
                        reminders if reminders is not None else baseline.get("reminders", [])
                    ),
                }
        return merged

    @staticmethod
    def default_role_playbook() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Built-in role directives keyed by environment and role."""

        return {
            "research": {
                "principal investigator": {
                    "planning_hints": ["synthesize_findings", "delegate_tasks"],
                    "reminders": [
                        "Drive the research agenda, delegate blockers, and keep everyone aligned.",
                        "Narrate key takeaways for the team after each discovery.",
                    ],
                },
                "field researcher": {
                    "planning_hints": ["collect_interviews", "prioritize_new_rooms"],
                    "reminders": [
                        "Keep interviewing participants and capture fresh observations.",
                        "Visit new rooms to surface untapped information.",
                    ],
                },
                "citation librarian": {
                    "planning_hints": ["track_citations", "verify_sources"],
                    "reminders": [
                        "Ensure every claim is backed by a citation and flag gaps early.",
                        "Share reference formats and keep a running bibliography for the team.",
                    ],
                },
                "report assembler": {
                    "planning_hints": ["draft_sections", "organize_notes"],
                    "reminders": [
                        "Continuously structure notes into report sections as facts arrive.",
                        "Call for missing sections before submission time.",
                    ],
                },
                "qa reviewer": {
                    "planning_hints": ["check_accuracy", "flag_gaps"],
                    "reminders": [
                        "Audit facts and tone, and highlight inconsistencies for the drafter.",
                        "Maintain a final review checklist so the team knows what remains.",
                    ],
                },
            },
            "policy": {
                "requirements lead": {
                    "planning_hints": ["gather_requirements", "clarify_constraints"],
                    "reminders": [
                        "Capture stakeholder needs and confirm constraints before drafting.",
                        "Summarize requirement gaps so the drafter can act quickly.",
                    ],
                },
                "field owner": {
                    "planning_hints": ["provide_examples", "surface_risks"],
                    "reminders": [
                        "Offer domain examples and call out real-world risks or blockers.",
                        "Keep the team honest about what will or will not work in practice.",
                    ],
                },
                "policy drafter": {
                    "planning_hints": ["write_policy_language", "resolve_ambiguities"],
                    "reminders": [
                        "Transform requirements into clear policy text and close open questions.",
                        "Coordinate with submitters to keep the policy ready for approval.",
                    ],
                },
                "compliance submitter": {
                    "planning_hints": ["monitor_submission_readiness", "collect_signoffs"],
                    "reminders": [
                        "Track paperwork status and remind teammates about missing approvals.",
                        "Keep forms, citations, and attestations ready for timely submission.",
                    ],
                },
                "qa auditor": {
                    "planning_hints": ["audit_gaps", "enforce_adherence"],
                    "reminders": [
                        "Review drafts against standards and flag compliance risks early.",
                        "Report readiness blockers so the submitter can clear them.",
                    ],
                },
            },
            "nav": {
                "route planner": {
                    "planning_hints": ["optimize_paths", "coordinate_routes"],
                    "reminders": [
                        "Share efficient routes and reroute teammates around congestion.",
                        "Keep navigation plans updated as rooms change.",
                    ],
                },
                "room scout": {
                    "planning_hints": ["prioritize_new_rooms", "spot_obstacles"],
                    "reminders": [
                        "Scout unexplored rooms first and report hazards immediately.",
                        "Surface opportunities that unblock the rest of the team.",
                    ],
                },
                "signal relay": {
                    "planning_hints": ["broadcast_navigation_updates", "sync_positions"],
                    "reminders": [
                        "Keep everyone informed of moves, discoveries, and detours.",
                        "Confirm positions when the team splits up to avoid confusion.",
                    ],
                },
                "data logger": {
                    "planning_hints": ["record_positions", "track_timing"],
                    "reminders": [
                        "Maintain accurate timestamps and room visit history.",
                        "Share logs so planners can optimize next routes.",
                    ],
                },
                "recovery/support": {
                    "planning_hints": ["help_stuck_agents", "coordinate_regroup"],
                    "reminders": [
                        "Monitor for stuck teammates and guide them back to productive paths.",
                        "Organize regroup points when the team drifts apart.",
                    ],
                },
            },
        }

    def _normalize_playbook(
        self, playbook: Dict[str, Dict[str, Dict[str, List[str]]]]
    ) -> Dict[str, Dict[str, RoleDirectives]]:
        normalized: Dict[str, Dict[str, RoleDirectives]] = {}
        for env_key, roles in (playbook or {}).items():
            if not isinstance(roles, dict):
                continue
            env_bucket: Dict[str, RoleDirectives] = {}
            for role, directives in roles.items():
                if not isinstance(directives, dict):
                    continue
                env_bucket[role.lower()] = RoleDirectives(
                    planning_hints=list(directives.get("planning_hints", [])),
                    reminders=list(directives.get("reminders", [])),
                )
            if env_bucket:
                normalized[env_key.lower()] = env_bucket
        return normalized

    def _role_directives(
        self, environment: Optional[str], role: Optional[str]
    ) -> RoleDirectives:
        env_key = (environment or "").lower()
        role_key = (role or "").lower()
        env_bundle = self.role_playbook.get(env_key) or self.role_playbook.get("default")
        if not env_bundle:
            return RoleDirectives()
        return env_bundle.get(role_key, RoleDirectives())


__all__ = ["AlignmentContext", "MetaOrchestrator", "RoleDirectives"]

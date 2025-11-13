"""Objective management logic for assigning goals to agents."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Callable, Dict, Optional
from uuid import uuid4

from schemas.logs import ActionLog
from schemas.objectives import DEFAULT_OBJECTIVE_TEMPLATES, Objective, ObjectiveTemplate

RewardCallback = Callable[[str, Dict[str, float], Objective], None]


class ObjectiveManager:
    """Assigns objectives and tracks progress from action logs."""

    def __init__(
        self,
        templates: Optional[Dict[str, ObjectiveTemplate]] = None,
        *,
        enabled: bool = True,
        seed: int = 0,
        reward_callback: Optional[RewardCallback] = None,
    ) -> None:
        self.enabled = enabled
        self.templates: Dict[str, ObjectiveTemplate] = templates or DEFAULT_OBJECTIVE_TEMPLATES
        self.reward_callback = reward_callback
        self.agent_objectives: Dict[str, Objective] = {}
        self.rng = random.Random(seed)

    # ---- lifecycle helpers ----

    def register_reward_callback(self, callback: RewardCallback) -> None:
        self.reward_callback = callback

    def current_objective(self, agent_id: str) -> Optional[Objective]:
        return self.agent_objectives.get(agent_id)

    def assign_objective(self, agent_id: str, template_name: Optional[str] = None) -> Optional[Objective]:
        if not self.enabled or not self.templates:
            return None
        template = self._select_template(template_name)
        if not template:
            return None
        objective = Objective(
            objective_id=f"{agent_id}-{uuid4().hex[:8]}",
            agent_id=agent_id,
            type=template.type,
            description=template.description,
            requirements=dict(template.requirements),
            progress={key: 0 for key in template.requirements},
            status="active",
            assigned_at=datetime.utcnow(),
            reward=dict(template.reward),
        )
        self.agent_objectives[agent_id] = objective
        return objective

    def ensure_objective(self, agent_id: str) -> Optional[Objective]:
        current = self.agent_objectives.get(agent_id)
        if current is None or current.status in {"completed", "failed"}:
            return self.assign_objective(agent_id)
        return current

    def process_action_log(self, action_log: ActionLog) -> Optional[Objective]:
        if not self.enabled:
            return None
        objective = self.agent_objectives.get(action_log.agent_id)
        if not objective or objective.status != "active":
            return None
        requirement_key = action_log.action_type
        if requirement_key not in objective.requirements:
            return None
        if action_log.outcome not in {"success", "noop"}:
            return None
        objective.progress[requirement_key] = min(
            objective.progress.get(requirement_key, 0) + 1,
            objective.requirements[requirement_key],
        )
        if objective.is_complete():
            objective.status = "completed"
            objective.completed_at = datetime.utcnow()
            if self.reward_callback and objective.reward:
                self.reward_callback(objective.agent_id, objective.reward, objective)
            return objective
        return None

    # ---- internal helpers ----

    def _select_template(self, template_name: Optional[str]) -> Optional[ObjectiveTemplate]:
        if template_name:
            return self.templates.get(template_name)
        if not self.templates:
            return None
        names = sorted(self.templates.keys())
        choice = self.rng.choice(names)
        return self.templates[choice]


__all__ = ["ObjectiveManager", "RewardCallback"]

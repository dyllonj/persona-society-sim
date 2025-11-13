"""Objective schema and reusable templates for agent tasking."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


ObjectiveStatusLiteral = Literal["pending", "active", "completed", "failed"]


class Objective(BaseModel):
    """Structured goal tracking for an agent."""

    objective_id: str
    agent_id: str
    type: str
    description: str
    requirements: Dict[str, int] = Field(default_factory=dict)
    progress: Dict[str, int] = Field(default_factory=dict)
    status: ObjectiveStatusLiteral = "pending"
    assigned_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    reward: Dict[str, float] = Field(default_factory=dict)

    def is_complete(self) -> bool:
        if not self.requirements:
            return False
        return all(
            self.progress.get(key, 0) >= required for key, required in self.requirements.items()
        )


class ObjectiveTemplate(BaseModel):
    """Reusable blueprint for populating objectives."""

    name: str
    type: str
    description: str
    requirements: Dict[str, int]
    reward: Dict[str, float] = Field(default_factory=dict)


DEFAULT_OBJECTIVE_TEMPLATES: Dict[str, ObjectiveTemplate] = {
    "socialize": ObjectiveTemplate(
        name="socialize",
        type="socialize",
        description="Start successful conversations with three different agents.",
        requirements={"talk": 3},
        reward={"satisfaction": 3.0},
    ),
    "work_project": ObjectiveTemplate(
        name="work_project",
        type="work",
        description="Complete two work actions to contribute to the town.",
        requirements={"work": 2},
        reward={"satisfaction": 2.0},
    ),
    "support_market": ObjectiveTemplate(
        name="support_market",
        type="trade",
        description="Participate in one trade and one gift exchange.",
        requirements={"trade": 1, "gift": 1},
        reward={"satisfaction": 2.5},
    ),
}


__all__ = [
    "Objective",
    "ObjectiveTemplate",
    "ObjectiveStatusLiteral",
    "DEFAULT_OBJECTIVE_TEMPLATES",
]

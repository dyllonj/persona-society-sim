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
    "policy_checklist": ObjectiveTemplate(
        name="policy_checklist",
        type="policy",
        description="Fill all required checklist fields and submit a compliant plan.",
        requirements={"submit_plan": 1},
        reward={"satisfaction": 3.0},
    ),
    "navigation_discovery": ObjectiveTemplate(
        name="navigation_discovery",
        type="navigation",
        description="Visit multiple locations, scan tokens, and coordinate to minimize overlap.",
        requirements={"move": 3, "scan": 3},
        reward={"satisfaction": 2.5},
    ),
    "research_facts": ObjectiveTemplate(
        name="research_facts",
        type="research",
        description="Collect facts from the corpus at the library and submit a report.",
        requirements={"submit_report": 1},
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
    "explore": ObjectiveTemplate(
        name="explore",
        type="explore",
        description="Visit three different locations to explore the town.",
        requirements={"move": 3},
        reward={"satisfaction": 2.0},
    ),
    "collaborate": ObjectiveTemplate(
        name="collaborate",
        type="collaborate",
        description="Work together with others on shared goals.",
        requirements={"talk": 2, "work": 1},
        reward={"satisfaction": 3.5},
    ),
    "gather": ObjectiveTemplate(
        name="gather",
        type="gather",
        description="Collect resources through trading.",
        requirements={"trade": 2},
        reward={"satisfaction": 2.0},
    ),
    "research": ObjectiveTemplate(
        name="research",
        type="research",
        description="Engage in focused study and knowledge sharing.",
        requirements={"talk": 2},
        reward={"satisfaction": 2.5},
    ),
    "community_builder": ObjectiveTemplate(
        name="community_builder",
        type="community",
        description="Foster community connections through diverse interactions.",
        requirements={"talk": 2, "gift": 1, "move": 1},
        reward={"satisfaction": 4.0},
    ),
}


__all__ = [
    "Objective",
    "ObjectiveTemplate",
    "ObjectiveStatusLiteral",
    "DEFAULT_OBJECTIVE_TEMPLATES",
]

"""Memory, reflection, and planning schemas inspired by Generative Agents."""

from __future__ import annotations

from datetime import datetime
from datetime import datetime
from typing import Dict, List, Literal, Optional

from utils.pydantic_compat import BaseModel, Field


MemoryKind = Literal["observation", "self_note", "reflection", "plan", "summary"]


class MemoryEvent(BaseModel):
    memory_id: str
    agent_id: str
    kind: MemoryKind
    tick: int
    timestamp: datetime
    text: str
    importance: float = 0.0
    recency_decay: float = 1.0
    recency_decay: float = 1.0
    embedding_id: Optional[str] = None
    source_msg_id: Optional[str] = None
    traits: Dict[str, float] = Field(default_factory=dict)


class Reflection(BaseModel):
    reflection_id: str
    agent_id: str
    tick: int
    text: str
    derived_implications: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    plan_id: str
    agent_id: str
    tick_start: int
    tick_end: int
    steps: List[str] = Field(default_factory=list)

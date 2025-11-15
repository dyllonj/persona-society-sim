"""Agent- and persona-related data schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from utils.pydantic_compat import BaseModel, Field


TraitLiteral = Literal["E", "A", "C", "O", "N", "TRUTH", "SYNC"]
MethodLiteral = Literal["CAA", "ActAdd"]
StatusLiteral = Literal["idle", "busy", "resting", "sanctioned"]


class SteeringVectorRef(BaseModel):
    trait: TraitLiteral
    method: MethodLiteral
    layer_ids: List[int]
    vector_store_id: str
    version: str


class PersonaCoeffs(BaseModel):
    E: float = Field(0.0, ge=-3.0, le=3.0)
    A: float = Field(0.0, ge=-3.0, le=3.0)
    C: float = Field(0.0, ge=-3.0, le=3.0)
    O: float = Field(0.0, ge=-3.0, le=3.0)
    N: float = Field(0.0, ge=-3.0, le=3.0)


class AgentState(BaseModel):
    agent_id: str
    display_name: str
    persona_coeffs: PersonaCoeffs
    steering_refs: List[SteeringVectorRef]
    active_alpha_overrides: Dict[str, float] = Field(default_factory=dict)
    system_prompt: str
    location_id: str
    status: StatusLiteral = "idle"
    goals: List[str] = Field(default_factory=list)
    created_at: datetime
    last_tick: int


class Rule(BaseModel):
    rule_id: str
    text: str
    proposer_id: Optional[str] = None
    enacted_at_tick: Optional[int] = None
    active: bool = False


class WorldState(BaseModel):
    tick: int
    time_of_day: Literal["morning", "afternoon", "evening", "night"]
    location_ids: List[str]
    active_rules: List[Rule]

"""Agent- and persona-related data schemas."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Dict, List, Literal, Optional

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
    """Bounded Big Five coefficients used for steering alphas."""

    TRAITS: ClassVar[List[str]] = ["E", "A", "C", "O", "N"]
    MIN_COEFF: ClassVar[float] = -1.0
    MAX_COEFF: ClassVar[float] = 1.0

    E: float = Field(0.0, ge=-1.0, le=1.0)
    A: float = Field(0.0, ge=-1.0, le=1.0)
    C: float = Field(0.0, ge=-1.0, le=1.0)
    O: float = Field(0.0, ge=-1.0, le=1.0)
    N: float = Field(0.0, ge=-1.0, le=1.0)

    def __init__(self, **data: float) -> None:  # type: ignore[override]
        normalized = self._normalize_payload(data)
        super().__init__(**normalized)

    @classmethod
    def _normalize_payload(cls, payload: Dict[str, float]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for trait in cls.TRAITS:
            raw_value = float(payload.get(trait, 0.0))
            normalized[trait] = cls._clamp_value(raw_value)
        for extra_key, extra_value in payload.items():
            if extra_key not in normalized:
                normalized[extra_key] = extra_value
        return normalized

    @classmethod
    def _clamp_value(cls, value: float) -> float:
        return max(cls.MIN_COEFF, min(cls.MAX_COEFF, value))


class AgentState(BaseModel):
    agent_id: str
    display_name: str
    persona_coeffs: PersonaCoeffs
    steering_refs: List[SteeringVectorRef]
    active_alpha_overrides: Dict[str, float] = Field(default_factory=dict)
    role: str = "generalist"
    role_description: Optional[str] = None
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
    priority: Literal["mandatory", "advisory"] = "mandatory"
    environment_tags: List[str] = Field(default_factory=list)


class WorldState(BaseModel):
    tick: int
    time_of_day: Literal["morning", "afternoon", "evening", "night"]
    location_ids: List[str]
    active_rules: List[Rule]

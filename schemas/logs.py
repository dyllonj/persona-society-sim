"""Interaction logs, graph snapshots, and safety telemetry schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from utils.pydantic_compat import BaseModel, Field

ChannelLiteral = Literal["public", "room", "direct", "group"]
ActionLiteral = Literal[
    "move",
    "talk",
    "trade",
    "work",
    "gift",
    "fill_field",
    "propose_plan",
    "submit_plan",
    "vote",
    "propose_rule",
    "enforce",
    "steal",
    "report",
    "rest",
    "join_group",
    "leave_group",
    "research",
    "cite",
    "submit_report",
    "scan",
]
OutcomeLiteral = Literal["success", "fail", "sanctioned", "noop"]
SanctionLiteral = Literal["fine", "ostracize", "ban", "warning"]
SafetyLiteral = Literal["toxicity", "hallucination", "leakage", "refusal", "policy-violation"]
SeverityLiteral = Literal["low", "medium", "high"]


class MsgLog(BaseModel):
    msg_id: str
    run_id: str
    tick: int
    channel: ChannelLiteral
    from_agent: str
    to_agent: Optional[str] = None
    room_id: Optional[str] = None
    content: str
    tokens_in: int
    tokens_out: int
    temperature: float
    top_p: float
    steering_snapshot: Dict[str, float]
    layers_used: List[int]


class ActionLog(BaseModel):
    action_id: str
    run_id: str
    tick: int
    agent_id: str
    action_type: ActionLiteral
    params: Dict[str, str] = Field(default_factory=dict)
    outcome: OutcomeLiteral
    info: Dict[str, str] = Field(default_factory=dict)
    prompt_text: Optional[str] = None
    prompt_hash: Optional[str] = None
    plan_metadata: Dict[str, Any] = Field(default_factory=dict)
    reflection_summary: Optional[str] = None
    reflection_implications: List[str] = Field(default_factory=list)


class EconTxn(BaseModel):
    txn_id: str
    run_id: str
    tick: int
    buyer_id: str
    seller_id: str
    item: str
    qty: int
    price: float


class VoteLog(BaseModel):
    vote_id: str
    run_id: str
    tick: int
    proposal_id: str
    agent_id: str
    vote: Literal["yes", "no", "abstain"]


class SanctionLog(BaseModel):
    sanction_id: str
    run_id: str
    tick: int
    actor_id: str
    target_id: str
    kind: SanctionLiteral
    justification: str
    amount: Optional[float] = None


class SafetyEvent(BaseModel):
    event_id: str
    run_id: str
    tick: int
    agent_id: str
    kind: SafetyLiteral
    severity: SeverityLiteral
    applied_alpha_delta: Dict[str, float] = Field(default_factory=dict)


class Edge(BaseModel):
    src: str
    dst: str
    weight: float
    kind: Literal["message", "trade", "sanction", "group"]


class GraphSnapshot(BaseModel):
    run_id: str
    tick: int
    edges: List[Edge]
    centrality: Dict[str, float] = Field(default_factory=dict)
    trait_key: Optional[str] = None
    band_metadata: Dict[str, object] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    run_id: str
    tick: int
    cooperation_rate: float
    gini_wealth: float
    polarization_modularity: float
    conflicts: int
    rule_enforcement_cost: float
    trait_key: Optional[str] = None
    band_metadata: Dict[str, object] = Field(default_factory=dict)

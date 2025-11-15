"""Run configuration schemas and steering vector metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from utils.pydantic_compat import BaseModel


class SteeringVectorEntry(BaseModel):
    vector_store_id: str
    trait: str
    method: str
    layer_id: int
    vector_path: str
    pos_set_hash: str
    neg_set_hash: str
    created_at: datetime


class RunConfig(BaseModel):
    run_id: str
    git_commit: str
    model_name: str
    layers: List[int]
    population: int
    steps: int
    scenario: str
    seed: int
    steering: Dict[str, float]
    notes: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime]
    tokens_in: int
    tokens_out: int
    crashes: int

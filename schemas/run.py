"""Run configuration schemas and steering vector metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from utils.pydantic_compat import BaseModel, Field


class SteeringVectorEntry(BaseModel):
    vector_store_id: str
    trait: str
    method: str
    layer_id: int
    vector_path: str
    train_set_hash: str
    eval_set_hash: Optional[str] = None
    created_at: datetime


class SteeringMetadataFiles(BaseModel):
    personas: Optional[str] = None
    vectors: Optional[str] = None
    traits: Dict[str, str] = Field(default_factory=dict)


class SteeringConfig(BaseModel):
    strength: float = 1.0
    coefficients: Dict[str, float] = Field(default_factory=dict)
    vector_norm: Dict[str, float] = Field(default_factory=dict)
    metadata_files: SteeringMetadataFiles = Field(default_factory=SteeringMetadataFiles)


class RunConfig(BaseModel):
    run_id: str
    git_commit: str
    model_name: str
    population: int
    steps: int
    scenario: str
    seed: int
    steering: SteeringConfig
    notes: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime]
    tokens_in: int
    tokens_out: int
    crashes: int

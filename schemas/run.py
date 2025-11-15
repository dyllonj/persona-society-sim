"""Run configuration schemas and steering vector metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

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


class LoggingConfig(BaseModel):
    db_url: Optional[str] = None
    parquet_dir: Optional[str] = None


class RunSafetyConfig(BaseModel):
    alpha_clip: float = 1.0
    toxicity_threshold: float = 0.4
    governor_backoff: float = 0.2


class InferenceConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 120


class OptimizationConfig(BaseModel):
    reflect_every_n_ticks: int = 1
    use_quantization: bool = False
    batch_size: Optional[int] = None


class ObjectivesConfig(BaseModel):
    enabled: bool = False
    templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class RunConfig(BaseModel):
    run_id: str
    git_commit: str
    model_name: str
    population: int
    steps: int
    scenario: str
    seed: int
    steering: SteeringConfig
    logging: Optional[LoggingConfig] = None
    safety: Optional[RunSafetyConfig] = None
    inference: Optional[InferenceConfig] = None
    optimization: Optional[OptimizationConfig] = None
    objectives: Optional[ObjectivesConfig] = None
    probes: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime]
    tokens_in: int
    tokens_out: int
    crashes: int

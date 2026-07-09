"""Compatibility layer to gracefully fall back when Pydantic isn't available."""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly via imports
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field

    if hasattr(_PydanticBaseModel, "model_config"):
        from pydantic import ConfigDict

        class BaseModel(_PydanticBaseModel):
            """Project base model with legitimate ``model_*`` telemetry fields."""

            model_config = ConfigDict(protected_namespaces=())

    else:  # pragma: no cover - Pydantic v1 compatibility
        BaseModel = _PydanticBaseModel
except ModuleNotFoundError:  # pragma: no cover - minimal fallback for tests
    from utils.pydantic_stub import BaseModel, Field  # type: ignore

__all__ = ["BaseModel", "Field"]

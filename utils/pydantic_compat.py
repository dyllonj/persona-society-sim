"""Compatibility layer to gracefully fall back when Pydantic isn't available."""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly via imports
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - minimal fallback for tests
    from utils.pydantic_stub import BaseModel, Field  # type: ignore

__all__ = ["BaseModel", "Field"]

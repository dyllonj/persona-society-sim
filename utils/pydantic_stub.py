"""Minimal subset of Pydantic features for test environments without the dependency."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime
from typing import Any, ClassVar, Dict, get_origin, get_type_hints

_UNSET = object()


class FieldInfo:
    def __init__(self, default: Any = _UNSET, default_factory=None, **_ignored: Any) -> None:
        self.default = default
        self.default_factory = default_factory


def Field(default: Any = _UNSET, *, default_factory=None, **kwargs: Any) -> FieldInfo:  # noqa: D401
    """Return lightweight metadata compatible with the BaseModel stub."""
    if default is not _UNSET and default_factory is not None:
        raise ValueError("Cannot specify both default and default_factory")
    return FieldInfo(default=default, default_factory=default_factory, **kwargs)


class BaseModel:
    """Very small subset of the Pydantic BaseModel API used in tests."""

    def __init__(self, **data: Any) -> None:
        annotations = _resolve_annotations(self.__class__)
        for name, annotation in annotations.items():
            if name.startswith("_"):
                continue
            if get_origin(annotation) is ClassVar:
                continue
            value = data.pop(name, _UNSET)
            if value is _UNSET:
                field_info = getattr(self.__class__, name, _UNSET)
                if isinstance(field_info, FieldInfo):
                    if field_info.default_factory is not None:
                        value = field_info.default_factory()
                    elif field_info.default is not _UNSET:
                        value = deepcopy(field_info.default)
                    else:
                        raise ValueError(f"Missing required field: {name}")
                elif field_info is not _UNSET:
                    value = deepcopy(field_info)
                else:
                    raise ValueError(f"Missing required field: {name}")
            setattr(self, name, value)
        for extra, value in data.items():
            setattr(self, extra, value)

    def model_dump(self) -> Dict[str, Any]:
        annotations = _resolve_annotations(self.__class__)
        result: Dict[str, Any] = {}
        for name, annotation in annotations.items():
            if name.startswith("_"):
                continue
            if get_origin(annotation) is ClassVar:
                continue
            value = getattr(self, name)
            if isinstance(value, BaseModel):
                result[name] = value.model_dump()
            else:
                result[name] = deepcopy(value)
        return result

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), default=_json_encoder)

    @classmethod
    def model_validate_json(cls, payload: str) -> "BaseModel":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("model_validate_json expects a JSON object")
        return cls(**data)


def _json_encoder(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


__all__ = ["BaseModel", "Field"]


def _resolve_annotations(cls) -> Dict[str, Any]:
    try:
        return get_type_hints(cls)
    except Exception:
        return getattr(cls, "__annotations__", {})

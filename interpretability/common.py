from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

JSON_FIELDS = {
    "input_ids",
    "attention_mask",
    "generated_ids",
    "effective_alphas",
    "steering_vector_ids",
    "steering_vector_hashes",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def load_prompts(path: Path, *, limit: int | None = None) -> list[str]:
    prompts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, str):
                text = payload
            elif isinstance(payload, dict):
                text = next(
                    (
                        payload[key]
                        for key in ("text", "prompt", "content")
                        if isinstance(payload.get(key), str)
                    ),
                    None,
                )
            else:
                text = None
            if not text:
                raise ValueError(f"{path}:{line_number} has no text/prompt/content string")
            prompts.append(text)
            if limit is not None and len(prompts) >= limit:
                break
    else:
        prompts = [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n")]
        prompts = [prompt for prompt in prompts if prompt]
        if limit is not None:
            prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts loaded from {path}")
    return prompts


def normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in JSON_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = json.loads(value)
    return normalized


def read_inference_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(normalize_event(json.loads(line)))
        elif path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            events.extend(normalize_event(row) for row in pq.read_table(path).to_pylist())
        else:
            raise ValueError(f"unsupported inference event format: {path}")
    return events


def resolve_event_paths(value: Path) -> list[Path]:
    if value.is_file():
        return [value]
    paths = sorted(value.rglob("*.parquet")) + sorted(value.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no Parquet or JSONL inference events under {value}")
    return paths

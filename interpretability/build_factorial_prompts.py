"""Build condition-blind structured-action prompts from held-out trait scenarios.

Only ``question_text`` enters model-visible text. Contrast options and their
labels remain inaccessible to generation; their source stratum is retained as
analysis metadata outside the prompt string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .common import sha256_file, sha256_json, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from common import sha256_file, sha256_json, write_json_atomic  # type: ignore


SCHEMA_VERSION = "factorial-prompts-1.0"
DEFAULT_ITEMS_PER_STRATUM = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGIN_FILES = {
    "E": "extraversion_eval.jsonl",
    "A": "agreeableness_eval.jsonl",
    "C": "conscientiousness_eval.jsonl",
}
FORBIDDEN_PROMPT_LABELS = re.compile(
    r"\b(?:"
    r"persona|personality|personality\s+trait|big\s+five|"
    r"extraversion|extravert(?:ed)?|extroversion|extrovert(?:ed)?|introvert(?:ed)?|"
    r"agreeableness|agreeable|disagreeable|"
    r"conscientiousness|conscientious|unconscientious|"
    r"high[-\s]+trait|low[-\s]+trait"
    r")\b",
    flags=re.IGNORECASE,
)

ACTION_PARAMETER_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "move": (("destination",), ("destination",)),
    "talk": ((), ("topic",)),
    "work": ((), ("task",)),
    "gift": (("recipient",), ("recipient", "item", "qty")),
    "scan": ((), ()),
    "fill_field": (("field_name", "value"), ("field_name", "value")),
    "propose_plan": ((), ("summary",)),
    "submit_plan": ((), ()),
    "research": ((), ("query", "doc_id")),
    "cite": ((), ("doc_id",)),
    "submit_report": ((), ()),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build neutral structured-action prompts from held-out eval scenarios"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "prompts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-stratum", type=int, default=DEFAULT_ITEMS_PER_STRATUM)
    return parser.parse_args()


def render_structured_prompt(scenario: str) -> str:
    """Render the scenario using the simulator's action JSON contract."""

    scenario = " ".join(scenario.split())
    if not scenario:
        raise ValueError("scenario text must not be empty")
    supported = "; ".join(
        f"{action}(params: {', '.join(allowed) or 'none'}; "
        f"required: {', '.join(required) or 'none'})"
        for action, (required, allowed) in ACTION_PARAMETER_SPEC.items()
    )
    prompt = (
        "System: You are an agent in a social simulation. Decide and perform exactly one "
        "action.\n"
        "Return ONLY one JSON object with keys action, params, and utterance.\n"
        f"Supported actions: {supported}.\n"
        "Use only parameters appropriate for the selected action. The utterance must be a "
        "concise first-person sentence. Select the action yourself from the supported list.\n\n"
        f"Scenario: {scenario}\n"
        "JSON:"
    )
    match = FORBIDDEN_PROMPT_LABELS.search(prompt)
    if match:
        raise ValueError(f"scenario contains forbidden condition cue {match.group(0)!r}")
    return prompt


def _load_eval_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        required = {
            "id",
            "question_text",
            "option_a",
            "option_b",
            "option_a_is_high",
            "option_b_is_high",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"{path}:{line_number} lacks fields {missing}")
        if bool(payload["option_a_is_high"]) == bool(payload["option_b_is_high"]):
            raise ValueError(f"{path}:{line_number} must mark exactly one high option")
        records.append(payload)
    return records


def _portable_path(path: Path) -> str:
    """Keep repository-owned provenance relocatable across machines."""

    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT))
    return str(resolved)


def build_prompt_records(
    sources: Mapping[str, Path],
    *,
    expected_per_stratum: int | None = DEFAULT_ITEMS_PER_STRATUM,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build stable, round-robin prompt records and source provenance."""

    if set(sources) != set(ORIGIN_FILES):
        raise ValueError(f"sources must contain exactly {sorted(ORIGIN_FILES)}")
    if expected_per_stratum is not None and expected_per_stratum <= 0:
        raise ValueError("expected_per_stratum must be positive")

    loaded: dict[str, list[dict[str, Any]]] = {}
    source_provenance: dict[str, Any] = {}
    source_ids: set[tuple[str, str]] = set()
    prompt_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for stratum in ORIGIN_FILES:
        path = Path(sources[stratum]).resolve()
        records = _load_eval_records(path)
        if expected_per_stratum is not None and len(records) != expected_per_stratum:
            raise ValueError(
                f"origin stratum {stratum} has {len(records)} records; "
                f"expected {expected_per_stratum}"
            )
        loaded[stratum] = records
        source_provenance[stratum] = {
            "path": _portable_path(path),
            "sha256": sha256_file(path),
            "records": len(records),
        }

    output: list[dict[str, Any]] = []
    max_records = max(len(records) for records in loaded.values())
    for within_index in range(max_records):
        for stratum in ORIGIN_FILES:
            if within_index >= len(loaded[stratum]):
                continue
            source = loaded[stratum][within_index]
            source_id = str(source["id"]).strip()
            if not source_id:
                raise ValueError(f"origin stratum {stratum} has an empty source id")
            identity = (stratum, source_id)
            if identity in source_ids:
                raise ValueError(f"duplicate source id in stratum {stratum}: {source_id}")
            source_ids.add(identity)

            scenario = " ".join(str(source["question_text"]).split())
            prompt = render_structured_prompt(scenario)
            for option_field in ("option_a", "option_b"):
                option = " ".join(str(source[option_field]).split())
                if option and option.casefold() in prompt.casefold():
                    raise ValueError(
                        f"source option leaked into prompt for {stratum}/{source_id}"
                    )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompt_id = "factorial-prompt-" + sha256_json(
                {"origin_stratum": stratum, "source_id": source_id, "scenario": scenario}
            )[:16]
            if prompt_id in prompt_ids:
                raise ValueError(f"stable prompt id collision: {prompt_id}")
            if prompt_hash in prompt_hashes:
                raise ValueError(f"duplicate model-visible prompt for {stratum}/{source_id}")
            prompt_ids.add(prompt_id)
            prompt_hashes.add(prompt_hash)
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "prompt_id": prompt_id,
                    "text": prompt,
                    "prompt_sha256": prompt_hash,
                    "origin_stratum": stratum,
                    "source_id": source_id,
                    "source_file": Path(sources[stratum]).name,
                    "source_record_index": within_index,
                }
            )

    provenance = {
        "sources": source_provenance,
        "template_sha256": hashlib.sha256(
            render_structured_prompt("<SCENARIO>").encode("utf-8")
        ).hexdigest(),
        "expected_per_stratum": expected_per_stratum,
    }
    return output, provenance


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prompt_bundle(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output.suffix.lower() != ".jsonl":
        raise ValueError("factorial prompt output must end in .jsonl")
    _write_jsonl_atomic(output, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "output": output.name,
        "output_sha256": sha256_file(output),
        "prompts": len(rows),
        "prompt_ids_sha256": sha256_json([row["prompt_id"] for row in rows]),
        "prompt_hashes_sha256": sha256_json([row["prompt_sha256"] for row in rows]),
        **dict(provenance),
        "script_sha256": sha256_file(Path(__file__)),
    }
    manifest["manifest_content_sha256"] = sha256_json(manifest)
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    sources = {stratum: args.eval_dir / filename for stratum, filename in ORIGIN_FILES.items()}
    rows, provenance = build_prompt_records(
        sources,
        expected_per_stratum=args.expected_per_stratum,
    )
    manifest = write_prompt_bundle(args.output, rows, provenance)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Validate and descriptively analyze paired live-generation factorial arms.

The analysis unit is a prompt/seed pair. This module intentionally does not
construct personality scores, call a model judge, or treat generations as
independent simulation replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

try:
    from .common import sha256_file, sha256_json
except ImportError:  # pragma: no cover - direct script execution
    from common import sha256_file, sha256_json  # type: ignore


SCHEMA_VERSION = "factorial-analysis-1.0"
CONDITION_ORDER = ("neutral", "E_only", "A_only", "C_only", "E_A_C", "placebo_shuffled")
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
        description="Validate and analyze a paired six-arm live-generation factorial"
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompt-metadata", type=Path, required=True)
    parser.add_argument("--rubric-scores", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _verify_content_hash(manifest: Mapping[str, Any], *, path: Path) -> None:
    expected = manifest.get("manifest_content_sha256")
    if not isinstance(expected, str):
        raise ValueError(f"{path} has no manifest_content_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    actual = sha256_json(unhashed)
    if actual != expected:
        raise ValueError(f"manifest content hash mismatch: manifest={expected}, actual={actual}")


def _load_factorial_manifest(path: Path, events_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _verify_content_hash(manifest, path=path)
    if manifest.get("schema_version") != "factorial-1.0":
        raise ValueError(f"unsupported factorial manifest schema: {manifest.get('schema_version')}")
    actual_output_hash = sha256_file(events_path)
    if actual_output_hash != manifest.get("output_sha256"):
        raise ValueError(
            f"factorial output hash mismatch: manifest={manifest.get('output_sha256')}, "
            f"actual={actual_output_hash}"
        )
    condition_names = tuple(
        condition.get("name") if isinstance(condition, dict) else condition
        for condition in manifest.get("conditions") or []
    )
    if condition_names != CONDITION_ORDER:
        raise ValueError(
            f"factorial conditions mismatch: expected={CONDITION_ORDER}, actual={condition_names}"
        )
    expected_events = int(manifest.get("prompts", 0)) * len(CONDITION_ORDER)
    if int(manifest.get("events", -1)) != expected_events:
        raise ValueError(
            f"manifest event count is not prompts × six: events={manifest.get('events')}, "
            f"prompts={manifest.get('prompts')}"
        )
    return manifest


def _load_prompt_metadata(
    path: Path,
    factorial_manifest: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError("prompt metadata is empty")
    if sha256_file(path) != factorial_manifest.get("prompt_source_sha256"):
        raise ValueError("prompt metadata hash does not match factorial prompt source hash")
    expected_hashes = factorial_manifest.get("prompt_hashes") or []
    actual_hashes = [
        hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()
        for row in rows
    ]
    if actual_hashes != expected_hashes:
        raise ValueError("prompt metadata order/content does not match factorial manifest")

    by_hash: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for index, (row, actual_hash) in enumerate(zip(rows, actual_hashes, strict=True)):
        prompt_id = row.get("prompt_id")
        stratum = row.get("origin_stratum")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompt metadata row {index} has no stable prompt_id")
        if prompt_id in ids:
            raise ValueError(f"duplicate stable prompt id: {prompt_id}")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"prompt metadata row {index} has no origin_stratum")
        if row.get("prompt_sha256") != actual_hash:
            raise ValueError(f"prompt metadata hash mismatch for {prompt_id}")
        if actual_hash in by_hash:
            raise ValueError(f"duplicate model-visible prompt hash: {actual_hash}")
        ids.add(prompt_id)
        by_hash[actual_hash] = row

    sidecar_path = path.with_suffix(".manifest.json")
    sidecar: dict[str, Any] | None = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        _verify_content_hash(sidecar, path=sidecar_path)
        if sidecar.get("output_sha256") != sha256_file(path):
            raise ValueError("prompt bundle sidecar output hash mismatch")
        if int(sidecar.get("prompts", -1)) != len(rows):
            raise ValueError("prompt bundle sidecar prompt count mismatch")
    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "manifest_path": str(sidecar_path.resolve()) if sidecar is not None else None,
        "manifest_sha256": sha256_file(sidecar_path) if sidecar is not None else None,
    }
    return by_hash, provenance


def parse_structured_action(completion: str) -> dict[str, Any]:
    """Parse using the simulator's structured-action acceptance contract."""

    result = {
        "json_syntax_valid": False,
        "structured_action_valid": False,
        "action": None,
        "params": None,
        "utterance": None,
        "error": None,
    }
    try:
        start = completion.index("{")
        payload, _ = json.JSONDecoder().raw_decode(completion[start:])
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON value must be an object")
        result["json_syntax_valid"] = True
        action = payload.get("action")
        if not isinstance(action, str) or action not in ACTION_PARAMETER_SPEC:
            raise ValueError(f"unsupported action: {action!r}")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        required, allowed = ACTION_PARAMETER_SPEC[action]
        unexpected = sorted(set(params) - set(allowed))
        if unexpected:
            raise ValueError(f"unexpected params for {action}: {unexpected}")
        normalized_params: dict[str, str] = {}
        for key, value in params.items():
            if isinstance(value, dict | list) or value is None:
                raise ValueError(f"param {key} must be a scalar")
            normalized_params[str(key)] = str(value)
        missing = [key for key in required if not normalized_params.get(key)]
        if missing:
            raise ValueError(f"missing params for {action}: {missing}")
        utterance = payload.get("utterance", "")
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError("utterance must be a non-empty string")
        result.update(
            structured_action_valid=True,
            action=action,
            params=normalized_params,
            utterance=utterance,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    return result


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _length_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": fmean(values),
        "median": median(values),
    }


def _validate_events(
    events: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    prompt_by_hash: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, dict[str, Any]]]]:
    if len(events) != int(manifest["events"]):
        raise ValueError(
            f"event row count mismatch: manifest={manifest['events']}, actual={len(events)}"
        )
    trace_ids: set[str] = set()
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    enriched: list[dict[str, Any]] = []
    manifest_hashes = manifest.get("prompt_hashes") or []
    manifest_ids = manifest.get("prompt_ids") or []
    condition_specs = {
        condition["name"]: condition
        for condition in manifest["conditions"]
        if isinstance(condition, dict)
    }
    if manifest_ids and len(manifest_ids) != len(manifest_hashes):
        raise ValueError("factorial manifest prompt_ids/prompt_hashes length mismatch")
    for row_index, event in enumerate(events):
        if event.get("schema_version") != "factorial-event-1.0":
            raise ValueError(f"event {row_index} has an unsupported schema")
        if event.get("run_id") != manifest.get("run_id"):
            raise ValueError(f"event {row_index} run_id does not match manifest")
        trace_id = event.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id or trace_id in trace_ids:
            raise ValueError(f"event {row_index} has a missing or duplicate trace_id")
        trace_ids.add(trace_id)
        condition = event.get("condition")
        if condition not in CONDITION_ORDER:
            raise ValueError(f"event {trace_id} has unknown condition {condition!r}")
        if event.get("condition_index") != CONDITION_ORDER.index(str(condition)):
            raise ValueError(f"event {trace_id} has an invalid condition_index")
        condition_spec = condition_specs[str(condition)]
        for field in ("effective_alphas", "controller_alphas", "vector_mode"):
            if field in condition_spec and event.get(field) != condition_spec[field]:
                raise ValueError(f"event {trace_id} {field} does not match factorial manifest")
        for event_field, manifest_field in (
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("inference_dtype", "dtype"),
            ("quantization", "quantization"),
            ("do_sample", "do_sample"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            if manifest_field in manifest and event.get(event_field) != manifest[manifest_field]:
                raise ValueError(
                    f"event {trace_id} {event_field} does not match factorial manifest"
                )
        prompt_id = event.get("prompt_id")
        paired_seed = event.get("paired_seed")
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(paired_seed, int):
            raise ValueError(f"event {trace_id} has invalid prompt_id/paired_seed")
        prompt_index = event.get("prompt_index")
        if not isinstance(prompt_index, int) or not 0 <= prompt_index < len(manifest_hashes):
            raise ValueError(f"event {trace_id} has invalid prompt_index")
        if manifest_ids and prompt_id != manifest_ids[prompt_index]:
            raise ValueError(f"event {trace_id} prompt_id does not match factorial manifest")
        prompt_text = event.get("prompt_text")
        if not isinstance(prompt_text, str):
            raise ValueError(f"event {trace_id} has no prompt_text")
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if event.get("prompt_hash") != prompt_hash or manifest_hashes[prompt_index] != prompt_hash:
            raise ValueError(f"event {trace_id} prompt hash mismatch")
        metadata = prompt_by_hash.get(prompt_hash)
        if metadata is None:
            raise ValueError(f"event {trace_id} has no origin-stratum prompt metadata")
        if manifest_ids and prompt_id != metadata["prompt_id"]:
            raise ValueError(f"event {trace_id} prompt_id does not match prompt metadata")
        for field in ("origin_stratum", "source_id"):
            if event.get(field) is not None and event.get(field) != metadata.get(field):
                raise ValueError(f"event {trace_id} {field} does not match prompt metadata")
        generated_ids = event.get("generated_ids")
        if (
            not isinstance(generated_ids, list)
            or any(not isinstance(token, int) for token in generated_ids)
            or event.get("generated_token_count") != len(generated_ids)
        ):
            raise ValueError(f"event {trace_id} has invalid generated token data")
        input_ids = event.get("input_ids")
        attention_mask = event.get("attention_mask")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(not isinstance(token, int) for token in input_ids)
            or not isinstance(attention_mask, list)
            or any(not isinstance(value, int) for value in attention_mask)
            or len(input_ids) != len(attention_mask)
            or event.get("prompt_token_count") != len(input_ids)
        ):
            raise ValueError(f"event {trace_id} has invalid prompt token data")
        completion = event.get("raw_completion")
        if not isinstance(completion, str):
            raise ValueError(f"event {trace_id} has no raw_completion")

        key = (prompt_id, paired_seed)
        if condition in groups[key]:
            raise ValueError(f"duplicate arm {condition} for pair {key}")
        analysis = {
            **event,
            "stable_prompt_id": metadata["prompt_id"],
            "origin_stratum": metadata["origin_stratum"],
            "source_id": metadata.get("source_id"),
            "completion_character_count": len(completion),
            **parse_structured_action(completion),
            "external_rubric_score": None,
        }
        groups[key][str(condition)] = analysis
        enriched.append(analysis)

    expected = set(CONDITION_ORDER)
    for key, arms in groups.items():
        if set(arms) != expected:
            raise ValueError(
                f"incomplete factorial pair {key}: missing={sorted(expected - arms.keys())}, "
                f"extra={sorted(arms.keys() - expected)}"
            )
        neutral = arms["neutral"]
        stable_id = neutral["stable_prompt_id"]
        prompt_hash = neutral["prompt_hash"]
        prompt_index = neutral["prompt_index"]
        for condition, event in arms.items():
            for field, expected_value in (
                ("stable_prompt_id", stable_id),
                ("prompt_hash", prompt_hash),
                ("prompt_index", prompt_index),
                ("prompt_text", neutral["prompt_text"]),
                ("input_ids", neutral.get("input_ids")),
                ("attention_mask", neutral.get("attention_mask")),
                ("model_id", neutral.get("model_id")),
                ("model_revision", neutral.get("model_revision")),
                ("tokenizer_revision", neutral.get("tokenizer_revision")),
            ):
                if event.get(field) != expected_value:
                    raise ValueError(f"pair {key} arm {condition} disagrees on {field}")
    if len(groups) != int(manifest["prompts"]):
        raise ValueError(
            f"paired prompt/seed count mismatch: manifest={manifest['prompts']}, "
            f"actual={len(groups)}"
        )
    return enriched, dict(groups)


def _load_rubric_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("rubric scores must be JSONL or CSV")


def _join_rubric_scores(
    path: Path,
    groups: Mapping[tuple[str, int], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    aliases: dict[tuple[str, str], dict[str, Any]] = {}
    for (event_prompt_id, _seed), arms in groups.items():
        for condition, event in arms.items():
            for prompt_id in (event_prompt_id, event["stable_prompt_id"]):
                key = (prompt_id, condition)
                if key in aliases and aliases[key] is not event:
                    raise ValueError(f"rubric prompt alias is ambiguous: {prompt_id}")
                aliases[key] = event

    seen_events: set[str] = set()
    rubric_ids: set[str] = set()
    rows = _load_rubric_rows(path)
    for row_index, row in enumerate(rows):
        prompt_id = str(row.get("prompt_id") or "")
        condition = str(row.get("condition") or "")
        event = aliases.get((prompt_id, condition))
        if event is None:
            raise ValueError(
                f"rubric row {row_index} does not match a prompt/condition: "
                f"{prompt_id}/{condition}"
            )
        if event["trace_id"] in seen_events:
            raise ValueError(
                f"multiple rubric scores for {event['stable_prompt_id']}/{condition}; "
                "analyze one prespecified rubric at a time"
            )
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rubric row {row_index} has no numeric score") from exc
        if not math.isfinite(score):
            raise ValueError(f"rubric row {row_index} score must be finite")
        event["external_rubric_score"] = score
        seen_events.add(event["trace_id"])
        rubric_ids.add(str(row.get("rubric_id") or "default"))
    if len(rubric_ids) > 1:
        raise ValueError("rubric file mixes rubric_id values; analyze one rubric at a time")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "rubric_id": next(iter(rubric_ids), "default"),
        "coverage": len(seen_events),
    }


def _arm_summary(
    rows: Sequence[dict[str, Any]],
    neutral_by_pair: Mapping[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "generations": 0,
            "json_syntax_valid_rate": None,
            "structured_action_valid_rate": None,
            "action_distribution": {},
            "generated_token_length": _length_summary([]),
            "completion_character_length": _length_summary([]),
            "exact_path_match_neutral_rate": None,
            "exact_path_divergence_neutral_rate": None,
            "external_rubric": {"available": 0, "mean": None},
        }
    syntax_valid = sum(bool(row["json_syntax_valid"]) for row in rows)
    structured_valid = sum(bool(row["structured_action_valid"]) for row in rows)
    actions = Counter(
        row["action"] if row["structured_action_valid"] else "__invalid__" for row in rows
    )
    exact_matches: list[bool] = []
    first_divergences: list[int] = []
    for row in rows:
        key = (row["prompt_id"], row["paired_seed"])
        neutral = neutral_by_pair[key]
        first = _first_divergence(neutral["generated_ids"], row["generated_ids"])
        exact_matches.append(first is None)
        if first is not None:
            first_divergences.append(first)
    rubric_scores = [
        float(row["external_rubric_score"])
        for row in rows
        if row["external_rubric_score"] is not None
    ]
    return {
        "generations": len(rows),
        "json_syntax_valid_rate": syntax_valid / len(rows),
        "structured_action_valid_rate": structured_valid / len(rows),
        "action_distribution": dict(sorted(actions.items())),
        "generated_token_length": _length_summary(
            [int(row["generated_token_count"]) for row in rows]
        ),
        "completion_character_length": _length_summary(
            [int(row["completion_character_count"]) for row in rows]
        ),
        "exact_path_match_neutral_rate": sum(exact_matches) / len(exact_matches),
        "exact_path_divergence_neutral_rate": 1.0
        - (sum(exact_matches) / len(exact_matches)),
        "first_divergence_token_index": _length_summary(first_divergences)
        if first_divergences
        else None,
        "external_rubric": {
            "available": len(rubric_scores),
            "mean": fmean(rubric_scores) if rubric_scores else None,
        },
    }


def _build_contrasts(
    groups: Mapping[tuple[str, int], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for (event_prompt_id, seed), arms in sorted(groups.items()):
        neutral = arms["neutral"]
        for condition in CONDITION_ORDER[1:]:
            treatment = arms[condition]
            first = _first_divergence(neutral["generated_ids"], treatment["generated_ids"])
            neutral_rubric = neutral["external_rubric_score"]
            treatment_rubric = treatment["external_rubric_score"]
            contrasts.append(
                {
                    "event_prompt_id": event_prompt_id,
                    "stable_prompt_id": neutral["stable_prompt_id"],
                    "source_id": neutral["source_id"],
                    "origin_stratum": neutral["origin_stratum"],
                    "paired_seed": seed,
                    "condition": condition,
                    "exact_generated_path_match": first is None,
                    "first_divergence_token_index": first,
                    "generated_token_count_delta": int(treatment["generated_token_count"])
                    - int(neutral["generated_token_count"]),
                    "completion_character_count_delta": int(
                        treatment["completion_character_count"]
                    )
                    - int(neutral["completion_character_count"]),
                    "json_syntax_validity_delta": int(treatment["json_syntax_valid"])
                    - int(neutral["json_syntax_valid"]),
                    "structured_action_validity_delta": int(
                        treatment["structured_action_valid"]
                    )
                    - int(neutral["structured_action_valid"]),
                    "neutral_action": neutral["action"],
                    "condition_action": treatment["action"],
                    "action_changed": (
                        treatment["action"] != neutral["action"]
                        if treatment["structured_action_valid"]
                        and neutral["structured_action_valid"]
                        else None
                    ),
                    "external_rubric_score_delta": (
                        float(treatment_rubric) - float(neutral_rubric)
                        if treatment_rubric is not None and neutral_rubric is not None
                        else None
                    ),
                }
            )
    return contrasts


def _contrast_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pairs": 0}
    comparable_actions = [row for row in rows if row["action_changed"] is not None]
    rubric_deltas = [
        row["external_rubric_score_delta"]
        for row in rows
        if row["external_rubric_score_delta"] is not None
    ]
    return {
        "pairs": len(rows),
        "exact_path_divergence_rate": sum(
            not row["exact_generated_path_match"] for row in rows
        )
        / len(rows),
        "mean_generated_token_count_delta": fmean(
            row["generated_token_count_delta"] for row in rows
        ),
        "mean_completion_character_count_delta": fmean(
            row["completion_character_count_delta"] for row in rows
        ),
        "mean_json_syntax_validity_delta": fmean(
            row["json_syntax_validity_delta"] for row in rows
        ),
        "mean_structured_action_validity_delta": fmean(
            row["structured_action_validity_delta"] for row in rows
        ),
        "action_change_rate_when_both_valid": (
            sum(bool(row["action_changed"]) for row in comparable_actions)
            / len(comparable_actions)
            if comparable_actions
            else None
        ),
        "action_pairs_compared": len(comparable_actions),
        "mean_external_rubric_score_delta": fmean(rubric_deltas)
        if rubric_deltas
        else None,
        "external_rubric_pairs": len(rubric_deltas),
    }


def build_analysis(
    *,
    events_path: Path,
    manifest_path: Path,
    prompt_metadata_path: Path,
    rubric_scores_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_factorial_manifest(manifest_path, events_path)
    prompt_by_hash, prompt_provenance = _load_prompt_metadata(
        prompt_metadata_path, manifest
    )
    events, groups = _validate_events(_load_jsonl(events_path), manifest, prompt_by_hash)
    rubric_provenance = (
        _join_rubric_scores(rubric_scores_path, groups) if rubric_scores_path else None
    )
    neutral_by_pair = {key: arms["neutral"] for key, arms in groups.items()}
    arms = {
        condition: _arm_summary(
            [event for event in events if event["condition"] == condition],
            neutral_by_pair,
        )
        for condition in CONDITION_ORDER
    }
    contrasts = _build_contrasts(groups)
    contrast_by_condition = {
        condition: _contrast_summary(
            [row for row in contrasts if row["condition"] == condition]
        )
        for condition in CONDITION_ORDER[1:]
    }

    strata: dict[str, Any] = {}
    stratum_names = sorted({event["origin_stratum"] for event in events})
    for stratum in stratum_names:
        stratum_events = [event for event in events if event["origin_stratum"] == stratum]
        stratum_pairs = {
            key: arms_for_pair
            for key, arms_for_pair in groups.items()
            if arms_for_pair["neutral"]["origin_stratum"] == stratum
        }
        stratum_neutral = {key: pair["neutral"] for key, pair in stratum_pairs.items()}
        stratum_contrasts = [row for row in contrasts if row["origin_stratum"] == stratum]
        strata[stratum] = {
            "prompt_seed_pairs": len(stratum_pairs),
            "arms": {
                condition: _arm_summary(
                    [event for event in stratum_events if event["condition"] == condition],
                    stratum_neutral,
                )
                for condition in CONDITION_ORDER
            },
            "paired_contrasts": {
                condition: _contrast_summary(
                    [row for row in stratum_contrasts if row["condition"] == condition]
                )
                for condition in CONDITION_ORDER[1:]
            },
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "analysis_unit": "paired prompt_id × seed",
            "evidence_level": "live-generation diagnostic, not independent simulation evidence",
            "personality_scores_constructed": False,
            "model_judge_calls": False,
            "inference": "descriptive paired contrasts only",
        },
        "provenance": {
            "events_path": str(events_path.resolve()),
            "events_sha256": sha256_file(events_path),
            "factorial_manifest_path": str(manifest_path.resolve()),
            "factorial_manifest_sha256": sha256_file(manifest_path),
            "run_id": manifest.get("run_id"),
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "prompt_metadata": prompt_provenance,
            "external_rubric": rubric_provenance,
        },
        "prompt_seed_pairs": len(groups),
        "generations": len(events),
        "conditions": list(CONDITION_ORDER),
        "arms": arms,
        "paired_contrasts": contrast_by_condition,
        "per_prompt_seed_contrasts": contrasts,
        "origin_strata": strata,
    }
    report["analysis_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "created_at"}
    )
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *(
            "| " + " | ".join(_fmt(cell).replace("|", "\\|") for cell in row) + " |"
            for row in rows
        ),
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Paired live-generation factorial analysis",
        "",
        "This is a descriptive paired-prompt diagnostic. The unit is a prompt/seed pair, "
        "not an independently replicated simulation. No personality score or model-judge "
        "score is constructed.",
        "",
        "## Provenance",
        "",
        *_table(
            ["Field", "Value"],
            [
                ["Run", report["provenance"]["run_id"]],
                ["Model", report["provenance"]["model_id"]],
                ["Revision", report["provenance"]["model_revision"]],
                ["Events SHA-256", report["provenance"]["events_sha256"]],
                ["Prompt/seed pairs", report["prompt_seed_pairs"]],
                ["Generations", report["generations"]],
                ["Analysis SHA-256", report["analysis_sha256"]],
            ],
        ),
        "",
        "## Arm summaries",
        "",
        *_table(
            [
                "Arm",
                "N",
                "JSON syntax valid",
                "Structured valid",
                "Mean tokens",
                "Divergence from neutral",
                "Actions",
            ],
            [
                [
                    condition,
                    summary["generations"],
                    summary["json_syntax_valid_rate"],
                    summary["structured_action_valid_rate"],
                    summary["generated_token_length"]["mean"],
                    summary["exact_path_divergence_neutral_rate"],
                    json.dumps(summary["action_distribution"], sort_keys=True),
                ]
                for condition, summary in report["arms"].items()
            ],
        ),
        "",
        "## Paired contrasts against neutral",
        "",
        *_table(
            [
                "Arm",
                "Pairs",
                "Path divergence",
                "Mean token Δ",
                "Action change (both valid)",
                "External rubric Δ",
            ],
            [
                [
                    condition,
                    summary["pairs"],
                    summary["exact_path_divergence_rate"],
                    summary["mean_generated_token_count_delta"],
                    summary["action_change_rate_when_both_valid"],
                    summary["mean_external_rubric_score_delta"],
                ]
                for condition, summary in report["paired_contrasts"].items()
            ],
        ),
        "",
        "## Origin-stratum summaries",
    ]
    for stratum, summary in report["origin_strata"].items():
        lines.extend(
            [
                "",
                f"### {stratum}",
                "",
                f"Prompt/seed pairs: {summary['prompt_seed_pairs']}.",
                "",
                *_table(
                    ["Arm", "Structured valid", "Mean tokens", "Divergence"],
                    [
                        [
                            condition,
                            arm["structured_action_valid_rate"],
                            arm["generated_token_length"]["mean"],
                            arm["exact_path_divergence_neutral_rate"],
                        ]
                        for condition, arm in summary["arms"].items()
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "The JSON companion contains every prompt/seed contrast, including first token "
            "divergence, action changes, validity deltas, length deltas, and any externally "
            "supplied rubric delta.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(prefix: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    if prefix.suffix.lower() in {".json", ".md"}:
        raise ValueError("--output-prefix must not include .json or .md")
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_temp = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    markdown_temp = markdown_path.with_name(f".{markdown_path.name}.tmp.{os.getpid()}")
    try:
        json_temp.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        markdown_temp.write_text(render_markdown(report), encoding="utf-8")
        json_temp.replace(json_path)
        markdown_temp.replace(markdown_path)
    finally:
        json_temp.unlink(missing_ok=True)
        markdown_temp.unlink(missing_ok=True)
    return json_path, markdown_path


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest or args.events.with_suffix(".manifest.json")
    report = build_analysis(
        events_path=args.events,
        manifest_path=manifest_path,
        prompt_metadata_path=args.prompt_metadata,
        rubric_scores_path=args.rubric_scores,
    )
    json_path, markdown_path = _write_outputs(args.output_prefix, report)
    print(
        json.dumps(
            {
                "analysis_sha256": report["analysis_sha256"],
                "json": str(json_path),
                "json_sha256": sha256_file(json_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

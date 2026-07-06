"""LLM judge helpers for generation-based persona evaluation.

The functions in this module are pure except for the ``JudgeClient`` protocol.
Tests and validation scripts can supply a static client without importing any
provider SDKs or making network calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


TRAIT_ALIASES: dict[str, str] = {
    "e": "E",
    "extraversion": "E",
    "extraverted": "E",
    "a": "A",
    "agreeableness": "A",
    "agreeable": "A",
    "c": "C",
    "conscientiousness": "C",
    "conscientious": "C",
    "o": "O",
    "openness": "O",
    "open": "O",
    "open-mindedness": "O",
    "n": "N",
    "neuroticism": "N",
    "neurotic": "N",
    "emotional_instability": "N",
}

TRAIT_NAMES: dict[str, str] = {
    "E": "extraversion",
    "A": "agreeableness",
    "C": "conscientiousness",
    "O": "openness",
    "N": "neuroticism",
}

DEFAULT_TRAITS: tuple[str, ...] = ("E", "A", "C", "O", "N")


class ModelFamilyError(ValueError):
    """Raised when judge/target model-family separation is violated."""


class LikertParseError(ValueError):
    """Raised when judge output does not contain valid 1-5 Likert scores."""


@dataclass(frozen=True)
class ModelFamilyPair:
    target_model: str
    judge_model: str
    target_family: str
    judge_family: str


@dataclass(frozen=True)
class JudgeRequest:
    prompt: str
    model: str
    system: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResponse:
    text: str
    model: str
    usage: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    scores: Mapping[str, int]
    response: JudgeResponse
    prompt: str
    family_pair: ModelFamilyPair | None = None


class JudgeClient(Protocol):
    """Provider-neutral client interface used by the judge helpers."""

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Return judge text for ``request``."""


class StaticJudgeClient:
    """Deterministic in-memory judge client for tests and validation scripts."""

    def __init__(self, responses: Sequence[str | JudgeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[JudgeRequest] = []

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("StaticJudgeClient has no responses left")
        response = self._responses.pop(0)
        if isinstance(response, JudgeResponse):
            return response
        return JudgeResponse(text=response, model=request.model)


def canonical_trait(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    if key not in TRAIT_ALIASES:
        raise ValueError(f"Unknown Big Five trait: {value}")
    return TRAIT_ALIASES[key]


def canonical_traits(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return DEFAULT_TRAITS
    return tuple(canonical_trait(value) for value in values)


def extract_model_family(model_name: str) -> str:
    """Extract a coarse model family suitable for judge independence checks."""

    normalized = model_name.strip().lower()
    if not normalized:
        raise ValueError("model_name must be non-empty")
    compact = re.sub(r"[^a-z0-9]+", "-", normalized)

    known_patterns = (
        ("llama", "llama"),
        ("qwen", "qwen"),
        ("mixtral", "mistral"),
        ("mistral", "mistral"),
        ("claude", "claude"),
        ("gemini", "gemini"),
        ("palm", "gemini"),
        ("gemma", "gemma"),
        ("deepseek", "deepseek"),
        ("command-r", "command-r"),
        ("grok", "grok"),
        ("gpt", "gpt"),
    )
    for marker, family in known_patterns:
        if marker in compact:
            return family
    if re.search(r"(?:^|-)o[134](?:-|$)", compact):
        return "openai-reasoning"

    parts = [part for part in compact.split("-") if part]
    if not parts:
        raise ValueError(f"Could not extract model family from {model_name!r}")
    provider_tail = normalized.rsplit("/", maxsplit=1)[-1]
    tail_parts = [part for part in re.split(r"[^a-z0-9]+", provider_tail) if part]
    return tail_parts[0] if tail_parts else parts[0]


def enforce_distinct_model_families(
    target_model: str,
    judge_model: str,
) -> ModelFamilyPair:
    """Require the judge model to come from a different coarse family."""

    pair = ModelFamilyPair(
        target_model=target_model,
        judge_model=judge_model,
        target_family=extract_model_family(target_model),
        judge_family=extract_model_family(judge_model),
    )
    if pair.target_family == pair.judge_family:
        raise ModelFamilyError(
            "Judge model must be from a different family than the target model: "
            f"{target_model!r} and {judge_model!r} both map to {pair.target_family!r}"
        )
    return pair


def _coerce_likert(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise LikertParseError(f"{label} is not a valid Likert score")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*([1-5])(?:\.0+)?(?:\s*/\s*5)?\s*", value)
        if not match:
            raise LikertParseError(f"{label} is not a valid Likert score")
        score = int(match.group(1))
    else:
        raise LikertParseError(f"{label} is not a valid Likert score")
    if score < 1 or score > 5:
        raise LikertParseError(f"{label}={score} is outside the 1-5 Likert range")
    return score


def _json_objects(text: str) -> list[Any]:
    stripped = text.strip()
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    parsed: list[Any] = []
    for candidate in candidates:
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return parsed


def _flatten_score_payload(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("scores", "trait_scores", "big_five"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def _parse_json_scores(text: str) -> dict[str, int]:
    for payload in _json_objects(text):
        mapping = _flatten_score_payload(payload)
        if mapping is None:
            continue
        scores: dict[str, int] = {}
        for key, value in mapping.items():
            try:
                code = canonical_trait(str(key))
            except ValueError:
                continue
            scores[code] = _coerce_likert(value, label=str(key))
        if scores:
            return scores
    return {}


def _parse_text_scores(text: str) -> dict[str, int]:
    trait_pattern = "|".join(
        sorted((re.escape(key) for key in TRAIT_ALIASES), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<![A-Za-z])(?P<trait>{trait_pattern})(?![A-Za-z])"
        r"\s*(?:score|rating)?\s*[:=\-]\s*"
        r"(?P<score>[1-5])(?:\.0+)?(?:\s*/\s*5)?",
        flags=re.IGNORECASE,
    )
    scores: dict[str, int] = {}
    for match in pattern.finditer(text):
        code = canonical_trait(match.group("trait"))
        scores[code] = _coerce_likert(match.group("score"), label=code)
    return scores


def parse_likert_scores(
    judge_output: str,
    expected_traits: Iterable[str] | None = None,
    *,
    require_all: bool = True,
) -> dict[str, int]:
    """Parse Big Five Likert scores from JSON or simple text judge output."""

    expected = canonical_traits(expected_traits) if expected_traits is not None else None
    scores = _parse_json_scores(judge_output) or _parse_text_scores(judge_output)
    if expected is not None:
        scores = {trait: scores[trait] for trait in expected if trait in scores}
        missing = [trait for trait in expected if trait not in scores]
        if missing and require_all:
            labels = ", ".join(missing)
            raise LikertParseError(f"Missing Likert score(s) for: {labels}")
    if not scores:
        raise LikertParseError("No Big Five Likert scores found in judge output")
    return scores


def build_trait_judge_prompt(
    generated_text: str,
    *,
    traits: Iterable[str] | None = None,
) -> str:
    """Build a compact judge prompt requesting JSON Big Five Likert scores."""

    codes = canonical_traits(traits)
    requested = ", ".join(f"{code} ({TRAIT_NAMES[code]})" for code in codes)
    return (
        "Rate the personality expressed by the generated text on 1-5 Likert "
        "scores, where 1 is very low expression and 5 is very high expression. "
        f"Return only JSON with keys for these traits: {requested}.\n\n"
        f"Generated text:\n{generated_text}"
    )


def score_text_with_judge(
    client: JudgeClient,
    generated_text: str,
    *,
    judge_model: str,
    target_model: str | None = None,
    traits: Iterable[str] | None = None,
) -> JudgeResult:
    """Ask a judge client to score generated text and parse its Likert output."""

    family_pair = (
        enforce_distinct_model_families(target_model, judge_model)
        if target_model is not None
        else None
    )
    prompt = build_trait_judge_prompt(generated_text, traits=traits)
    response = client.complete(JudgeRequest(prompt=prompt, model=judge_model))
    scores = parse_likert_scores(response.text, traits)
    return JudgeResult(
        scores=scores,
        response=response,
        prompt=prompt,
        family_pair=family_pair,
    )

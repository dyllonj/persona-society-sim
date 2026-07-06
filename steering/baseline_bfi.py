"""BFI-44 baseline scoring utilities for persona evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


SCALE_MIN = 1
SCALE_MAX = 5


@dataclass(frozen=True)
class BFIItem:
    number: int
    text: str
    trait: str
    reverse: bool = False


@dataclass(frozen=True)
class TraitHeadroom:
    trait: str
    score: float
    upward: float
    downward: float
    centered_alpha: float


@dataclass(frozen=True)
class BFIProfile:
    scores: Mapping[str, float]
    item_counts: Mapping[str, int]
    missing_items: tuple[int, ...]
    headroom: Mapping[str, TraitHeadroom]

    def to_dict(self) -> dict[str, object]:
        return {
            "scores": dict(self.scores),
            "item_counts": dict(self.item_counts),
            "missing_items": list(self.missing_items),
            "headroom": {
                trait: asdict(value) for trait, value in self.headroom.items()
            },
        }


TRAIT_ORDER: tuple[str, ...] = ("E", "A", "C", "O", "N")
TRAIT_LABELS: dict[str, str] = {
    "E": "extraversion",
    "A": "agreeableness",
    "C": "conscientiousness",
    "O": "openness",
    "N": "neuroticism",
}


BFI44_ITEMS: tuple[BFIItem, ...] = (
    BFIItem(1, "Is talkative", "E"),
    BFIItem(2, "Tends to find fault with others", "A", reverse=True),
    BFIItem(3, "Does a thorough job", "C"),
    BFIItem(4, "Is depressed, blue", "N"),
    BFIItem(5, "Is original, comes up with new ideas", "O"),
    BFIItem(6, "Is reserved", "E", reverse=True),
    BFIItem(7, "Is helpful and unselfish with others", "A"),
    BFIItem(8, "Can be somewhat careless", "C", reverse=True),
    BFIItem(9, "Is relaxed, handles stress well", "N", reverse=True),
    BFIItem(10, "Is curious about many different things", "O"),
    BFIItem(11, "Is full of energy", "E"),
    BFIItem(12, "Starts quarrels with others", "A", reverse=True),
    BFIItem(13, "Is a reliable worker", "C"),
    BFIItem(14, "Can be tense", "N"),
    BFIItem(15, "Is ingenious, a deep thinker", "O"),
    BFIItem(16, "Generates a lot of enthusiasm", "E"),
    BFIItem(17, "Has a forgiving nature", "A"),
    BFIItem(18, "Tends to be disorganized", "C", reverse=True),
    BFIItem(19, "Worries a lot", "N"),
    BFIItem(20, "Has an active imagination", "O"),
    BFIItem(21, "Tends to be quiet", "E", reverse=True),
    BFIItem(22, "Is generally trusting", "A"),
    BFIItem(23, "Tends to be lazy", "C", reverse=True),
    BFIItem(24, "Is emotionally stable, not easily upset", "N", reverse=True),
    BFIItem(25, "Is inventive", "O"),
    BFIItem(26, "Has an assertive personality", "E"),
    BFIItem(27, "Can be cold and aloof", "A", reverse=True),
    BFIItem(28, "Perseveres until the task is finished", "C"),
    BFIItem(29, "Can be moody", "N"),
    BFIItem(30, "Values artistic, aesthetic experiences", "O"),
    BFIItem(31, "Is sometimes shy, inhibited", "E", reverse=True),
    BFIItem(32, "Is considerate and kind to almost everyone", "A"),
    BFIItem(33, "Does things efficiently", "C"),
    BFIItem(34, "Remains calm in tense situations", "N", reverse=True),
    BFIItem(35, "Prefers work that is routine", "O", reverse=True),
    BFIItem(36, "Is outgoing, sociable", "E"),
    BFIItem(37, "Is sometimes rude to others", "A", reverse=True),
    BFIItem(38, "Makes plans and follows through with them", "C"),
    BFIItem(39, "Gets nervous easily", "N"),
    BFIItem(40, "Likes to reflect, play with ideas", "O"),
    BFIItem(41, "Has few artistic interests", "O", reverse=True),
    BFIItem(42, "Likes to cooperate with others", "A"),
    BFIItem(43, "Is easily distracted", "C", reverse=True),
    BFIItem(44, "Is sophisticated in art, music, or literature", "O"),
)


ITEM_BY_NUMBER: dict[int, BFIItem] = {item.number: item for item in BFI44_ITEMS}


def reverse_score(value: int, *, scale_min: int = SCALE_MIN, scale_max: int = SCALE_MAX) -> int:
    if value < scale_min or value > scale_max:
        raise ValueError(f"Likert response {value} is outside {scale_min}-{scale_max}")
    return scale_min + scale_max - value


def normalize_responses(responses: Mapping[int | str, int | str]) -> dict[int, int]:
    """Normalize item-response keys and values to integer BFI item ids/scores."""

    normalized: dict[int, int] = {}
    for raw_key, raw_value in responses.items():
        key_text = str(raw_key).strip().lower()
        if key_text.startswith("item_"):
            key_text = key_text[5:]
        elif key_text.startswith("q"):
            key_text = key_text[1:]
        try:
            item_number = int(key_text)
            score = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid BFI response entry: {raw_key!r}={raw_value!r}") from exc
        if item_number not in ITEM_BY_NUMBER:
            raise ValueError(f"Unknown BFI-44 item number: {item_number}")
        if score < SCALE_MIN or score > SCALE_MAX:
            raise ValueError(f"Item {item_number} response {score} is outside 1-5")
        normalized[item_number] = score
    return normalized


def score_bfi44(
    responses: Mapping[int | str, int | str],
    *,
    require_complete: bool = True,
) -> BFIProfile:
    """Score BFI-44 item responses into Big Five trait means and headroom."""

    normalized = normalize_responses(responses)
    missing = tuple(
        item.number for item in BFI44_ITEMS if item.number not in normalized
    )
    if missing and require_complete:
        raise ValueError(f"Missing BFI-44 item responses: {missing}")

    buckets: dict[str, list[int]] = {trait: [] for trait in TRAIT_ORDER}
    for item in BFI44_ITEMS:
        if item.number not in normalized:
            continue
        value = normalized[item.number]
        scored = reverse_score(value) if item.reverse else value
        buckets[item.trait].append(scored)

    scores: dict[str, float] = {}
    item_counts: dict[str, int] = {}
    for trait in TRAIT_ORDER:
        values = buckets[trait]
        item_counts[trait] = len(values)
        scores[trait] = round(sum(values) / len(values), 3) if values else 0.0

    headroom = profile_headroom(scores)
    return BFIProfile(
        scores=scores,
        item_counts=item_counts,
        missing_items=missing,
        headroom=headroom,
    )


def profile_headroom(scores: Mapping[str, float]) -> dict[str, TraitHeadroom]:
    """Compute upward/downward headroom and a [-1, 1] centered trait alpha."""

    midpoint = (SCALE_MIN + SCALE_MAX) / 2.0
    half_range = (SCALE_MAX - SCALE_MIN) / 2.0
    output: dict[str, TraitHeadroom] = {}
    for trait in TRAIT_ORDER:
        score = float(scores.get(trait, 0.0))
        upward = max(0.0, SCALE_MAX - score)
        downward = max(0.0, score - SCALE_MIN)
        centered = (score - midpoint) / half_range if score else 0.0
        output[trait] = TraitHeadroom(
            trait=trait,
            score=round(score, 3),
            upward=round(upward, 3),
            downward=round(downward, 3),
            centered_alpha=round(centered, 3),
        )
    return output


def build_bfi44_prompt(subject_text: str | None = None) -> str:
    """Build a questionnaire prompt for a model or human rater."""

    lines = [
        "Rate the subject on each BFI-44 item from 1 to 5.",
        "Return JSON with keys item_1 through item_44 and integer values.",
        "Scale: 1=disagree strongly, 3=neither, 5=agree strongly.",
    ]
    if subject_text:
        lines.extend(["", "Subject text:", subject_text])
    lines.append("")
    lines.append("Items:")
    for item in BFI44_ITEMS:
        lines.append(f"{item.number}. {item.text}")
    return "\n".join(lines)


def load_responses(path: Path | str) -> dict[int | str, int | str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("BFI response file must contain a JSON object")
    return dict(payload)


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BFI-44 baseline scoring utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score", help="Score a JSON file of BFI-44 responses")
    score_parser.add_argument("responses", type=Path)
    score_parser.add_argument("--allow-partial", action="store_true")

    prompt_parser = sub.add_parser("prompt", help="Print the BFI-44 prompt skeleton")
    prompt_parser.add_argument("--subject-text")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "score":
        profile = score_bfi44(
            load_responses(args.responses),
            require_complete=not args.allow_partial,
        )
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "prompt":
        print(build_bfi44_prompt(args.subject_text))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())

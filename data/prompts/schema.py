"""Prompt schema utilities for persona trait prompts.

This module contains helpers that describe the multiple-choice schema used
throughout ``data/prompts`` and a small CLI that can convert the legacy
``situation``/``positive``/``negative`` format into the new A/B structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class PromptItem:
    """Dataclass describing a single prompt in the A/B schema."""

    id: str
    question_text: str
    option_a: str
    option_b: str
    option_a_is_high: bool
    option_b_is_high: bool

    def validate(self) -> None:
        """Ensure the record satisfies the schema constraints."""

        if not self.id:
            raise ValueError("Prompt id must be a non-empty string")
        if not self.question_text:
            raise ValueError(f"Prompt {self.id} is missing a question_text")
        if not self.option_a or not self.option_b:
            raise ValueError(f"Prompt {self.id} requires two answer options")

        highs = int(self.option_a_is_high) + int(self.option_b_is_high)
        if highs != 1:
            raise ValueError(
                f"Prompt {self.id} must label exactly one high-trait option"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question_text": self.question_text,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "option_a_is_high": self.option_a_is_high,
            "option_b_is_high": self.option_b_is_high,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PromptItem":
        item = cls(
            id=payload["id"],
            question_text=payload["question_text"],
            option_a=payload["option_a"],
            option_b=payload["option_b"],
            option_a_is_high=payload["option_a_is_high"],
            option_b_is_high=payload["option_b_is_high"],
        )
        item.validate()
        return item


def load_prompt_items(path: Path | str) -> List[PromptItem]:
    path = Path(path)
    items: List[PromptItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            items.append(PromptItem.from_dict(data))
    return items


def _convert_legacy_record(payload: dict, high_first: bool = True) -> PromptItem:
    """Convert an old prompt dict into the new schema."""

    positive = payload["positive"].strip()
    negative = payload["negative"].strip()

    if high_first:
        option_a, option_b = positive, negative
        a_high, b_high = True, False
    else:
        option_a, option_b = negative, positive
        a_high, b_high = False, True

    item = PromptItem(
        id=payload["id"],
        question_text=payload["situation"].strip(),
        option_a=option_a,
        option_b=option_b,
        option_a_is_high=a_high,
        option_b_is_high=b_high,
    )
    item.validate()
    return item


def convert_legacy_file(
    source: Path | str,
    destination: Path | str,
    *,
    swap_options: bool = False,
) -> Sequence[PromptItem]:
    """Convert a JSONL file from the legacy schema into the new schema."""

    source = Path(source)
    destination = Path(destination)
    converted: List[PromptItem] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                legacy = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on {source}:{line_number}") from exc
            converted.append(_convert_legacy_record(legacy, high_first=not swap_options))

    with destination.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    return converted


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt conversion/validation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    convert_parser = sub.add_parser(
        "convert", help="Convert legacy situation/positive/negative prompts"
    )
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("destination", type=Path)
    convert_parser.add_argument(
        "--swap-options",
        action="store_true",
        help="Place the legacy negative option first (option A)",
    )

    validate_parser = sub.add_parser("validate", help="Validate A/B prompt files")
    validate_parser.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "convert":
        convert_legacy_file(
            args.source, args.destination, swap_options=args.swap_options
        )
        return 0
    if args.command == "validate":
        for path in args.paths:
            load_prompt_items(path)
        return 0
    return 1  # pragma: no cover - defensive


if __name__ == "__main__":
    raise SystemExit(_cli())

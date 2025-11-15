"""Tests for the multiple-choice prompt schema in ``data/prompts``."""

from pathlib import Path

from data.prompts.schema import load_prompt_items


PROMPT_DIR = Path(__file__).resolve().parents[1] / "data" / "prompts"


def test_prompt_files_have_consistent_stems_and_high_labels() -> None:
    for prompt_file in sorted(PROMPT_DIR.glob("*.jsonl")):
        items = load_prompt_items(prompt_file)
        assert items, f"{prompt_file} is empty"

        seen_stems: dict[str, str] = {}
        for item in items:
            # Confirm identical stems when the same id appears multiple times.
            previous = seen_stems.setdefault(item.id, item.question_text)
            assert (
                previous == item.question_text
            ), f"Conflicting stem for {item.id} in {prompt_file}"

            assert (
                item.option_a_is_high ^ item.option_b_is_high
            ), f"{item.id} should have exactly one high-trait option"

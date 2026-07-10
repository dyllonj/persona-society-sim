import json
from pathlib import Path

import pytest

from data.prompts.schema import PromptItem, load_prompt_items
from scripts.split_eval_prompts import (
    normalize_prompt_text,
    split_prompt_file,
    split_prompt_items,
    verify_disjoint,
)


def _item(idx: int, *, prompt_id: str | None = None) -> PromptItem:
    return PromptItem(
        id=prompt_id or f"T{idx}",
        question_text=f"Scenario {idx}",
        option_a=f"High behavior {idx}",
        option_b=f"Low behavior {idx}",
        option_a_is_high=True,
        option_b_is_high=False,
    )


def _write_jsonl(path: Path, items: list[PromptItem]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict()) + "\n")


def test_split_prompt_items_keeps_duplicate_ids_in_one_split() -> None:
    items = [_item(1, prompt_id="shared"), _item(2, prompt_id="shared")]
    items.extend(_item(idx) for idx in range(3, 8))

    train, eval_records = split_prompt_items(items, eval_count=2, seed=1)

    verification = verify_disjoint(train, eval_records)
    assert verification.ok
    train_ids = {item.id for item in train}
    eval_ids = {item.id for item in eval_records}
    assert "shared" not in (train_ids & eval_ids)


def test_split_prompt_file_writes_disjoint_outputs(tmp_path: Path) -> None:
    source = tmp_path / "extraversion.jsonl"
    _write_jsonl(source, [_item(idx) for idx in range(1, 7)])

    result = split_prompt_file(source, eval_count=2, seed=7)

    train = load_prompt_items(result.train_path)
    eval_records = load_prompt_items(result.eval_path)
    assert result.train_count == 4
    assert result.eval_count == 2
    assert result.verification.ok
    assert verify_disjoint(train, eval_records).ok


def test_verify_disjoint_catches_overlapping_ids_and_content() -> None:
    train = [_item(1, prompt_id="same")]
    eval_records = [_item(1, prompt_id="same")]

    verification = verify_disjoint(train, eval_records)

    assert not verification.ok
    assert verification.overlapping_ids == ("same",)
    assert verification.overlapping_fingerprints
    assert verification.overlapping_normalized_texts


def test_verify_disjoint_catches_reformatted_and_near_duplicate_text() -> None:
    train = [_item(1, prompt_id="train")]
    train[0].question_text = "A teammate carefully reviews the final project report."
    eval_records = [_item(2, prompt_id="eval")]
    eval_records[0].question_text = "A teammate carefully reviews the final project report again."
    eval_records[0].option_a = "  HIGH behavior 1!! "

    verification = verify_disjoint(train, eval_records)

    assert not verification.ok
    assert normalize_prompt_text(eval_records[0].option_a) == "high behavior 1"
    assert verification.overlapping_normalized_texts == ("high behavior 1",)
    assert verification.near_duplicate_questions == ("train~eval:0.889",)


def test_split_prompt_file_refuses_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "agreeableness.jsonl"
    _write_jsonl(source, [_item(idx) for idx in range(1, 5)])
    (tmp_path / "agreeableness_train.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError):
        split_prompt_file(source, eval_count=1)

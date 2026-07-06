import json
from pathlib import Path

from steering.baseline_bfi import (
    BFI44_ITEMS,
    _cli,
    build_bfi44_prompt,
    profile_headroom,
    reverse_score,
    score_bfi44,
)


def _neutral_responses() -> dict[int, int]:
    return {item.number: 3 for item in BFI44_ITEMS}


def test_reverse_score_uses_five_point_scale() -> None:
    assert reverse_score(1) == 5
    assert reverse_score(3) == 3
    assert reverse_score(5) == 1


def test_score_bfi44_applies_reverse_coding_and_trait_means() -> None:
    responses = _neutral_responses()
    responses[1] = 5
    responses[6] = 1
    profile = score_bfi44(responses)

    assert profile.item_counts["E"] == 8
    assert profile.missing_items == ()
    assert profile.scores["E"] > 3.0
    assert profile.scores["A"] == 3.0


def test_profile_headroom_reports_directional_room_and_centered_alpha() -> None:
    headroom = profile_headroom({"E": 4.5, "A": 2.0, "C": 3.0, "O": 5.0, "N": 1.0})

    assert headroom["E"].upward == 0.5
    assert headroom["E"].downward == 3.5
    assert headroom["A"].centered_alpha == -0.5
    assert headroom["O"].upward == 0.0


def test_build_bfi44_prompt_contains_all_items() -> None:
    prompt = build_bfi44_prompt("sample transcript")

    assert "item_1 through item_44" in prompt
    assert "sample transcript" in prompt
    assert "44. Is sophisticated in art, music, or literature" in prompt


def test_bfi_cli_scores_json_without_model_calls(tmp_path: Path, capsys) -> None:
    response_file = tmp_path / "responses.json"
    response_file.write_text(json.dumps(_neutral_responses()), encoding="utf-8")

    exit_code = _cli(["score", str(response_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"scores"' in captured.out
    assert '"E": 3.0' in captured.out

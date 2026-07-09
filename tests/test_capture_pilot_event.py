from __future__ import annotations

import pytest

from interpretability.capture_pilot_event import _parse_alphas, _selected_action


def test_parse_pilot_alphas_requires_explicit_nonzero_trait_values():
    assert _parse_alphas("E=0.8,A=-0.5") == {"E": 0.8, "A": -0.5}

    with pytest.raises(ValueError, match="at least one nonzero"):
        _parse_alphas("E=0")


def test_selected_action_records_model_or_explicit_fallback():
    assert _selected_action('{"action":"research","params":{},"utterance":"I check."}') == (
        "research",
        "model",
        None,
    )

    action, source, error = _selected_action("not json")
    assert action == "talk"
    assert source == "planner_fallback"
    assert error

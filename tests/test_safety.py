from __future__ import annotations

import pytest

from safety.governor import SafetyConfig, SafetyGovernor


def test_safety_governor_flags_banned_phrase():
    governor = SafetyGovernor(
        SafetyConfig(alpha_clip=1.0, governor_backoff=0.5, banned_phrases=["forbidden"])
    )
    event = governor.evaluate(
        run_id="run",
        agent_id="agent-1",
        text="This contains a forbidden topic.",
        tick=1,
        current_alphas={"E": 1.0},
    )
    assert event is not None
    assert "E" in event.applied_alpha_delta


def test_safety_governor_respects_global_strength_clamp():
    governor = SafetyGovernor(
        SafetyConfig(alpha_clip=0.5, governor_backoff=0.0, global_alpha_strength=2.0)
    )
    assert pytest.approx(0.25) == governor.clamp(0.5)

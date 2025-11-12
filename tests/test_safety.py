from __future__ import annotations

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

import json
from pathlib import Path

import pytest

from metrics.tracker import MetricTracker
from schemas.logs import ActionLog


def _action(agent_id: str, tick: int, action_type: str) -> ActionLog:
    return ActionLog(
        action_id=f"{agent_id}-{tick}-{action_type}",
        run_id="unit",
        tick=tick,
        agent_id=agent_id,
        action_type=action_type,
        params={},
        outcome="success",
        info={},
    )


def _summary(tmp_path: Path, run_id: str) -> dict:
    log_path = tmp_path / f"run_{run_id}.jsonl"
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return lines[0]["summary"]


def test_behavioral_variance_empty_and_one_agent() -> None:
    empty = MetricTracker("empty")
    assert empty.behavioral_variance() == {
        "agent_count": 0,
        "action_distributions": {},
        "by_action": {},
        "overall": 0.0,
        "variance_vs_mean_ratio": {},
    }

    one_agent = MetricTracker("one-agent")
    one_agent.on_action(_action("agent-1", 1, "talk"))

    behavior = one_agent.behavioral_variance()
    assert behavior["agent_count"] == 1
    assert behavior["action_distributions"]["agent-1"] == {"talk": 1.0}
    assert behavior["by_action"] == {"talk": 0.0}
    assert behavior["variance_vs_mean_ratio"] == {"talk": 0.0}


def test_behavioral_variance_uses_per_agent_action_distributions(tmp_path: Path) -> None:
    tracker = MetricTracker("behavior", out_dir=tmp_path)
    tracker.on_action(_action("agent-1", 1, "talk"))
    tracker.on_action(_action("agent-1", 2, "talk"))
    tracker.on_action(_action("agent-2", 1, "research"))
    tracker.on_action(_action("agent-2", 2, "cite"))

    behavior = tracker.behavioral_variance()
    assert behavior["action_distributions"] == {
        "agent-1": {"talk": 1.0},
        "agent-2": {"cite": 0.5, "research": 0.5},
    }
    assert behavior["by_action"]["talk"] == pytest.approx(0.25)
    assert behavior["by_action"]["cite"] == pytest.approx(0.0625)
    assert behavior["by_action"]["research"] == pytest.approx(0.0625)
    assert behavior["overall"] == pytest.approx(0.125)
    assert behavior["variance_vs_mean_ratio"]["talk"] == pytest.approx(0.5)
    assert behavior["variance_vs_mean_ratio"]["cite"] == pytest.approx(0.25)

    tracker.flush()

    passive_behavior = _summary(tmp_path, "behavior")["passive_validity"]["behavioral_variance"]
    assert passive_behavior["action_distributions"] == behavior["action_distributions"]
    assert passive_behavior["by_action"]["talk"] == pytest.approx(0.25)

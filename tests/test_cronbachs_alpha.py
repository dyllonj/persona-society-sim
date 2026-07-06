import pytest

from metrics.tracker import MetricTracker
from schemas.logs import PersonaStabilityLog


def _stability(
    agent_id: str,
    probe_text: str,
    score: float,
    tick: int = 1,
) -> PersonaStabilityLog:
    return PersonaStabilityLog(
        agent_id=agent_id,
        tick=tick,
        probe_text=probe_text,
        embedding_distance_from_baseline=0.0,
        trait_scores={"A": score},
    )


def test_cronbachs_alpha_empty_and_single_item_are_undefined() -> None:
    empty = MetricTracker("empty")
    assert empty.cronbachs_alpha() == {}

    single_item = MetricTracker("single-item")
    single_item.on_persona_stability(_stability("agent-1", "q1", 3.0))
    single_item.on_persona_stability(_stability("agent-2", "q1", 4.0))

    assert single_item.cronbachs_alpha() == {"A": None}


def test_cronbachs_alpha_for_complete_probe_item_matrix() -> None:
    tracker = MetricTracker("alpha")
    tracker.on_persona_stability(_stability("agent-1", "q1", 1.0))
    tracker.on_persona_stability(_stability("agent-1", "q2", 2.0))
    tracker.on_persona_stability(_stability("agent-2", "q1", 3.0))
    tracker.on_persona_stability(_stability("agent-2", "q2", 4.0))

    assert tracker.cronbachs_alpha()["A"] == pytest.approx(1.0)


def test_cronbachs_alpha_averages_repeated_probe_items() -> None:
    tracker = MetricTracker("alpha-repeated")
    tracker.on_persona_stability(_stability("agent-1", "q1", 1.0, tick=1))
    tracker.on_persona_stability(_stability("agent-1", "q1", 3.0, tick=2))
    tracker.on_persona_stability(_stability("agent-1", "q2", 2.0))
    tracker.on_persona_stability(_stability("agent-2", "q1", 3.0))
    tracker.on_persona_stability(_stability("agent-2", "q2", 4.0))

    assert tracker.cronbachs_alpha()["A"] == pytest.approx(0.888889)

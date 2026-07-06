import json
from pathlib import Path

import pytest

from metrics.tracker import MetricTracker
from orchestrator.probes import PersonaStabilityProbeDefinition, ProbeManager
from schemas.logs import PersonaStabilityLog


def _stability(
    agent_id: str,
    tick: int,
    probe_text: str,
    trait_scores: dict[str, float],
    distance: float = 0.0,
) -> PersonaStabilityLog:
    return PersonaStabilityLog(
        agent_id=agent_id,
        tick=tick,
        probe_text=probe_text,
        embedding_distance_from_baseline=distance,
        trait_scores=trait_scores,
    )


def _summary(tmp_path: Path, run_id: str) -> dict:
    log_path = tmp_path / f"run_{run_id}.jsonl"
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return lines[0]["summary"]


def test_passive_validity_empty_data_sidecar(tmp_path: Path) -> None:
    tracker = MetricTracker("empty", out_dir=tmp_path)

    assert tracker.population_trait_variance() == {}
    assert tracker.cronbachs_alpha() == {}
    assert tracker.test_retest_stability() == {}
    assert tracker.behavioral_variance()["by_action"] == {}

    tracker.flush()

    passive = _summary(tmp_path, "empty")["passive_validity"]
    assert passive["samples"] == 0
    assert passive["population_trait_variance"] == {}
    assert passive["cronbachs_alpha"] == {}
    assert passive["test_retest_stability"] == {}
    assert passive["embedding_distance_from_baseline"] == {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
    }


def test_one_agent_repeated_probe_is_stable_with_zero_population_variance() -> None:
    tracker = MetricTracker("one-agent")
    tracker.on_persona_stability(_stability("agent-1", 1, "q1", {"A": 3.0}, 0.1))
    tracker.on_persona_stability(_stability("agent-1", 4, "q1", {"A": 3.0}, 0.2))

    assert tracker.population_trait_variance() == {"A": 0.0}
    assert tracker.test_retest_stability() == {"A": 1.0}
    assert tracker.cronbachs_alpha() == {"A": None}


def test_repeated_probe_records_feed_stability_variance_and_sidecar(tmp_path: Path) -> None:
    tracker = MetricTracker("stability", out_dir=tmp_path)
    tracker.on_persona_stability(_stability("agent-1", 1, "q1", {"A": 2.0}, 0.1))
    tracker.on_persona_stability(_stability("agent-1", 3, "q1", {"A": 2.5}, 0.2))
    tracker.on_persona_stability(_stability("agent-2", 1, "q1", {"A": 4.0}, 0.3))
    tracker.on_persona_stability(_stability("agent-2", 3, "q1", {"A": 4.5}, 0.4))

    assert tracker.population_trait_variance()["A"] == pytest.approx(1.0)
    assert tracker.test_retest_stability()["A"] == pytest.approx(1.0)

    tracker.flush()

    passive = _summary(tmp_path, "stability")["passive_validity"]
    assert passive["samples"] == 4
    assert passive["population_trait_variance"]["A"] == pytest.approx(1.0)
    assert passive["test_retest_stability"]["A"] == pytest.approx(1.0)
    assert passive["embedding_distance_from_baseline"]["mean"] == pytest.approx(0.25)


def test_persona_stability_probe_records_baseline_distance_and_trait_scores() -> None:
    definition = PersonaStabilityProbeDefinition(
        probe_id="self-description",
        prompt="Describe your current persona.",
        cadence=20,
        trait_keywords={"A": ["help", "share"], "C": ["plan", "task"]},
    )
    manager = ProbeManager([], [], [definition], persona_stability_interval=20, seed=0)

    assignment = manager.assign_probe("agent-1", tick=0)
    assert assignment is not None
    assert assignment.kind == "persona_stability"
    assert "current persona" in assignment.inject("Base observation")

    first = manager.record_persona_stability_response(
        "agent-1",
        assignment,
        0,
        "I help others, share updates, and plan each task carefully.",
    )
    second = manager.record_persona_stability_response(
        "agent-1",
        assignment,
        20,
        "I avoid people and improvise without a plan.",
    )

    assert first.embedding_distance_from_baseline == 0.0
    assert first.trait_scores["A"] == pytest.approx(1.0)
    assert first.trait_scores["C"] == pytest.approx(1.0)
    assert second.embedding_distance_from_baseline > 0.0

import pytest

pytest.importorskip("numpy")

from eval.probes import ProbeManager


def test_probe_manager_scores_likert():
    config = {
        "enabled": True,
        "questionnaires": {
            "probe_id": "sr",
            "cadence": 1,
            "start_tick": 0,
            "questions": [
                {"item_id": "E1", "trait": "E", "text": "I am talkative."},
                {"item_id": "A1", "trait": "A", "text": "I help others."},
            ],
        },
    }
    manager = ProbeManager(["agent-000"], config)
    probes = manager.tick(0)
    assert probes, "Expected a scheduled probe"
    scores = manager.score_probe(probes[0], raw_text="5 1")
    assert scores["E"] > scores["A"]

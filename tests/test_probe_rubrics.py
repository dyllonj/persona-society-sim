from schemas.logs import ActionLog, BehaviorProbeLog
from metrics.probe_rubrics import ProbeRubricScorer


def make_action(**kwargs):
    defaults = {
        "action_id": "a1",
        "run_id": "r1",
        "tick": 0,
        "agent_id": "agent-1",
        "action_type": "gift",
        "params": {},
        "outcome": "success",
        "info": {},
    }
    defaults.update(kwargs)
    return ActionLog(**defaults)


def make_behavior(**kwargs):
    defaults = {
        "log_id": "b1",
        "run_id": "r1",
        "tick": 0,
        "agent_id": "agent-1",
        "probe_id": "p1",
        "scenario": "",
        "prompt_text": "",
        "response_text": "",
    }
    defaults.update(kwargs)
    return BehaviorProbeLog(**defaults)


def test_action_rubric_scores():
    scorer = ProbeRubricScorer()
    gift = make_action(
        action_type="gift",
        info={"trait_key": "A:high", "trait_band": "high"},
    )
    talk = make_action(
        action_id="a2",
        action_type="talk",
        info={"trait_key": "E:low", "trait_band": "low"},
    )
    explore = make_action(
        action_id="a3",
        action_type="move",
        params={"destination": "market"},
        info={"trait_key": "O:high", "trait_band": "high"},
    )
    repeat_explore = make_action(
        action_id="a4",
        action_type="move",
        params={"destination": "market"},
        info={"trait_key": "O:high", "trait_band": "high"},
    )

    scorer.ingest_actions([gift, talk, explore, repeat_explore])

    summary = scorer.summary()
    assert summary["by_trait"]["A"]["generosity"] == 1.0
    assert summary["by_trait"]["E"]["outreach"] > 0
    # Only the first exploration to a new destination should count
    assert summary["by_trait"]["O"]["exploration"] == 0.25


def test_behavior_probe_scoring():
    scorer = ProbeRubricScorer()
    positive = make_behavior(
        outcome="share",
        trait="A",
        affordance="generosity",
        preferred_outcome="share",
    )
    negative = make_behavior(
        log_id="b2",
        outcome="refuse",
        trait="A",
        affordance="generosity",
        preferred_outcome="share",
    )
    scorer.ingest_behavior_probes([positive, negative])
    summary = scorer.summary()

    assert summary["by_trait"]["A"]["generosity"] == 0.0
    assert summary["by_cohort"]["A"]["probe_total"] == 0.0

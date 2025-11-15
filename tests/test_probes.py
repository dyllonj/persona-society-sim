from orchestrator.probes import BehaviorProbeDefinition, LikertProbeDefinition, ProbeManager


def test_probe_manager_injects_and_scores_likert():
    definition = LikertProbeDefinition(
        probe_id="focus",
        question="Rate focus 1-5",
        trait="C",
        instructions="Respond with 1-5",
        cadence=1,
    )
    manager = ProbeManager([definition], [], likert_interval=1, behavior_interval=1, seed=0)
    assignment = manager.assign_probe("agent-001", tick=0)
    assert assignment is not None
    assert assignment.kind == "likert"
    injected = assignment.inject("Base observation")
    assert "Rate focus" in injected
    score, hint = ProbeManager.score_likert_response("I would say 4 because I feel sharp.")
    assert score == 4
    assert hint == "numeric"
    manager.complete_probe("agent-001", assignment, tick=0)


def test_probe_manager_scores_behavior_keywords():
    definition = BehaviorProbeDefinition(
        probe_id="supply",
        scenario="Give supplies?",
        instructions="Reply SHARE or REFUSE",
        outcomes={"share": ["share", "give"], "refuse": ["refuse"]},
        cadence=1,
    )
    manager = ProbeManager([], [definition], likert_interval=1, behavior_interval=1, seed=0)
    assignment = manager.assign_probe("agent-002", tick=0)
    assert assignment and assignment.kind == "behavior"
    response, hint = ProbeManager.score_behavior_response(assignment, "I would definitely share the supplies.")
    assert response == "share"
    assert hint == "share"
    manager.complete_probe("agent-002", assignment, tick=0)

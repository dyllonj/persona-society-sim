from orchestrator.probes import BehaviorProbeDefinition, ProbeManager


def test_behavior_probe_definition_instantiates_with_optional_fields():
    definition = BehaviorProbeDefinition(
        probe_id="share",
        scenario="Share supplies?",
        instructions="Respond SHARE or REFUSE",
        outcomes={"share": ["share"], "refuse": ["refuse"]},
        cadence=1,
    )

    assert definition.probe_id == "share"
    assert definition.trait is None
    assert definition.affordance is None
    assert definition.preferred_outcome is None


def test_probe_manager_imports_after_behavior_probe_dataclass_definition():
    manager = ProbeManager([], [], likert_interval=1, behavior_interval=1, seed=0)

    assert manager.likert_probes == []
    assert manager.behavior_probes == []

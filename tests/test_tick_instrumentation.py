from __future__ import annotations

from metrics.tick_instrumentation import TickInstrumentation


def test_tick_instrumentation_records_edges_and_macros():
    instrumentation = TickInstrumentation()
    instrumentation.on_tick_start(0)
    instrumentation.record_action(
        agent_id="agent-1",
        action_type="talk",
        success=True,
        params={"utterance": "hello"},
        info={"note": "chat"},
        steering_snapshot={"E": 0.5},
        persona_coeffs={"E": 2.0, "A": 0.0, "C": 0.0, "O": 0.0, "N": 0.0},
        encounter_room="library",
        encounter_participants=("agent-1", "agent-2"),
        satisfaction=0.25,
    )

    graph_inputs = instrumentation.graph_inputs()
    assert graph_inputs, "edges should be recorded"
    assert graph_inputs[0].edges[0].kind == "message"

    wealth_snapshot = {"agent-1": {"credits": 2}, "agent-2": {"credits": 1}}
    opinions = {"agent-1": 0.25, "agent-2": 0.0}
    macros = instrumentation.macro_inputs(wealth_snapshot, opinions)
    assert macros, "macro inputs should exist"
    macro = macros[0]
    assert macro.wealth["agent-1"] == 2.0
    assert macro.opinions["agent-1"] == 0.25

from __future__ import annotations

import math

from metrics.tick_instrumentation import TickInstrumentation
from metrics.social_dynamics import build_metrics_snapshot


def _record_action(instrumentation: TickInstrumentation, action_type: str, agent_id: str) -> None:
    instrumentation.record_action(
        agent_id=agent_id,
        action_type=action_type,
        success=True,
        params={},
        info={},
        steering_snapshot={},
        persona_coeffs={"E": 0.0, "A": 0.0, "C": 0.0, "O": 0.0, "N": 0.0},
        encounter_room="library",
        encounter_participants=(agent_id,),
        satisfaction=0.0,
    )


def test_action_type_entropy_tracks_tick_distribution():
    instrumentation = TickInstrumentation()
    instrumentation.on_tick_start(0)

    _record_action(instrumentation, "talk", "agent-1")
    _record_action(instrumentation, "talk", "agent-2")
    _record_action(instrumentation, "move", "agent-3")
    _record_action(instrumentation, "work", "agent-4")

    assert math.isclose(instrumentation.action_type_entropy(), 1.5)

    macros = instrumentation.macro_inputs(
        {
            "agent-1": {"credits": 1},
            "agent-2": {"credits": 1},
            "agent-3": {"credits": 1},
            "agent-4": {"credits": 1},
        },
        opinions={},
    )
    global_macro = next(macro for macro in macros if macro.trait_key is None)
    assert math.isclose(global_macro.action_type_entropy, 1.5)

    snapshot = build_metrics_snapshot(
        "run-entropy",
        0,
        global_macro.cooperation_events,
        global_macro.wealth,
        global_macro.opinions,
        global_macro.conflicts,
        global_macro.enforcement_cost,
        action_type_entropy=global_macro.action_type_entropy,
    )
    assert math.isclose(snapshot.action_type_entropy, 1.5)


def test_action_type_entropy_resets_at_tick_start():
    instrumentation = TickInstrumentation()
    instrumentation.on_tick_start(0)
    _record_action(instrumentation, "talk", "agent-1")
    _record_action(instrumentation, "move", "agent-2")

    assert math.isclose(instrumentation.action_type_entropy(), 1.0)

    instrumentation.on_tick_start(1)

    assert instrumentation.action_type_entropy() == 0.0

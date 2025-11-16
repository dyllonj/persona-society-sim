from __future__ import annotations

from metrics.tick_instrumentation import TickInstrumentation
from orchestrator.runner import SimulationRunner


class _DummyWorld:
    def __init__(self) -> None:
        self.tick = 0


class _DummyScheduler:
    pass


class _DummyLogSink:
    parquet_dir = None


class _LoggerStub:
    def __init__(self) -> None:
        self.enabled = True
        self.warnings: list[str] = []

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)


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
        prompt_hash="hash-1",
        plan_metadata={"action_type": "talk", "params": {}, "utterance": "say hi"},
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


def test_prompt_duplication_warning_emitted_for_tick_zero():
    logger = _LoggerStub()
    runner = SimulationRunner(
        run_id="run-1",
        world=_DummyWorld(),
        scheduler=_DummyScheduler(),
        agents=[],
        log_sink=_DummyLogSink(),
        temperature=0.7,
        top_p=1.0,
        console_logger=logger,
    )
    instrumentation = runner.tick_instrumentation
    instrumentation.on_tick_start(0)

    def _record(prompt_hash: str, prompt_text: str) -> None:
        instrumentation.record_action(
            agent_id="agent-1",
            action_type="talk",
            success=True,
            params={},
            info={},
            steering_snapshot={},
            persona_coeffs={"E": 0.0, "A": 0.0, "C": 0.0, "O": 0.0, "N": 0.0},
            encounter_room="library",
            encounter_participants=("agent-1", "agent-2"),
            satisfaction=0.0,
            prompt_hash=prompt_hash,
            prompt_text=prompt_text,
            plan_metadata=None,
        )

    for _ in range(3):
        _record("dup-hash", "Duplicate prompt text for warning test.")
    _record("unique-1", "Unique prompt one.")
    _record("unique-2", "Unique prompt two.")

    runner._warn_prompt_duplication(0)

    assert len(logger.warnings) == 1
    warning = logger.warnings[0]
    assert "Tick 0 prompt duplication rate" in warning
    assert "Sample prompt" in warning


def test_failed_trade_records_failure_without_edges_or_cooperation():
    instrumentation = TickInstrumentation()
    instrumentation.on_tick_start(0)

    instrumentation.record_action(
        agent_id="agent-1",
        action_type="trade",
        success=False,
        params={"recipient": "agent-2"},
        info={},
        steering_snapshot={},
        persona_coeffs={"E": 0.0, "A": 0.0, "C": 0.0, "O": 0.0, "N": 0.0},
        encounter_room="town_square",
        encounter_participants=("agent-1", "agent-2"),
        satisfaction=0.0,
        prompt_hash=None,
        plan_metadata=None,
    )

    assert instrumentation.graph_inputs() == []

    macros = instrumentation.macro_inputs(
        {"agent-1": {"credits": 1}, "agent-2": {"credits": 0}},
        opinions={},
    )
    assert any(m.trade_failures == 1 for m in macros)
    assert all(not m.cooperation_events for m in macros)

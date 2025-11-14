import json
from pathlib import Path

import pytest

for dependency in ("numpy", "torch", "pydantic", "yaml", "transformers", "pyarrow"):
    pytest.importorskip(dependency)

from orchestrator.cli import build_agents, build_language_backend, build_objective_manager
from orchestrator.console_logger import ConsoleLogger
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Scheduler
from safety.governor import SafetyConfig, SafetyGovernor
from storage.log_sink import LogSink
from env.world import World


class SpyLogSink(LogSink):
    def __init__(self, run_id: str):
        super().__init__(run_id, db_url=None, parquet_dir=None)
        self.recorded_actions = []
        self.recorded_messages = []

    def log_action(self, log):  # type: ignore[override]
        self.recorded_actions.append(log)
        super().log_action(log)

    def log_message(self, log):  # type: ignore[override]
        self.recorded_messages.append(log)
        super().log_message(log)


def _build_test_config():
    return {
        "run_id": "mock-sim",
        "population": 2,
        "steps": 3,
        "seed": 3,
        "steering": {"E": 0.1, "A": 0.2, "C": 0.3, "O": 0.0, "N": -0.2},
        "inference": {"temperature": 0.1, "top_p": 0.95, "max_new_tokens": 16},
        "optimization": {"reflect_every_n_ticks": 1},
        "objectives": {
            "enabled": True,
            "templates": {
                "unit_research": {
                    "type": "research",
                    "description": "Collect a single fact",
                    "requirements": {"research": 1},
                    "reward": {"satisfaction": 0.25},
                }
            },
        },
    }


def _action_snapshot(logs):
    return [
        {
            "tick": log.tick,
            "agent_id": log.agent_id,
            "action_type": log.action_type,
            "params": log.params,
            "outcome": log.outcome,
            "info": log.info,
        }
        for log in sorted(logs, key=lambda entry: (entry.tick, entry.agent_id))
    ]


def _message_snapshot(logs):
    return [
        {
            "tick": log.tick,
            "agent_id": log.from_agent,
            "room_id": log.room_id,
            "content": log.content,
        }
        for log in sorted(logs, key=lambda entry: (entry.tick, entry.from_agent))
    ]


def test_mock_simulation_reaches_objective(tmp_path):
    config = _build_test_config()
    world = World(data_dir="tests/data")
    world.configure_environment("research", difficulty=1)
    scheduler = Scheduler(world, seed=config["seed"])
    backend = build_language_backend(config, [], {}, mock=True)
    safety = SafetyGovernor(SafetyConfig(alpha_clip=1.5, toxicity_threshold=1.0, governor_backoff=0.1))
    agents = build_agents(config["run_id"], config, world, backend, safety)
    for agent in agents:
        world.move_agent(agent.state.agent_id, "library")
        agent.state.location_id = "library"
    objective_manager = build_objective_manager(config, "research", difficulty=1)
    spy_sink = SpyLogSink(config["run_id"])
    runner = SimulationRunner(
        run_id=config["run_id"],
        world=world,
        scheduler=scheduler,
        agents=agents,
        log_sink=spy_sink,
        temperature=config["inference"]["temperature"],
        top_p=config["inference"]["top_p"],
        console_logger=ConsoleLogger(enabled=False),
        objective_manager=objective_manager,
    )
    completed = []

    def reward_proxy(agent_id, reward, objective):
        completed.append((agent_id, objective.objective_id))
        runner._handle_objective_reward(agent_id, reward, objective)

    objective_manager.register_reward_callback(reward_proxy)
    runner.metric_tracker.out_dir = tmp_path

    runner.run(config["steps"], max_events_per_tick=4)

    assert completed, "ObjectiveManager never completed an objective"
    assert spy_sink.recorded_actions, "LogSink should capture action logs"
    assert spy_sink.recorded_messages, "LogSink should capture message logs"
    assert runner.metric_tracker.tick_collab_ratio, "MetricTracker did not record ratios"

    snapshot_path = Path("tests/data/mock_run_snapshot.json")
    expected = json.loads(snapshot_path.read_text())

    observed = {
        "actions": _action_snapshot(spy_sink.recorded_actions),
        "messages": _message_snapshot(spy_sink.recorded_messages),
    }

    assert observed == expected

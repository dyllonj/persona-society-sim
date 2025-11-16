import json
from pathlib import Path

import pytest

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.agent import Agent
from agents.language_backend import GenerationResult, LanguageBackend
from agents.memory import MemoryStore
from agents.planner import Planner
from agents.retrieval import MemoryRetriever
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs, Rule, SteeringVectorRef
from storage.log_sink import LogSink

REQUIRED_DEPENDENCIES = ("numpy", "torch", "pydantic", "yaml", "transformers", "pyarrow")
MISSING_DEPENDENCIES = [
    dep for dep in REQUIRED_DEPENDENCIES if importlib.util.find_spec(dep) is None
]
MISSING_DEP_REASON = (
    "Missing dependencies: " + ", ".join(MISSING_DEPENDENCIES)
    if MISSING_DEPENDENCIES
    else ""
)


class _StubLanguageBackend(LanguageBackend):
    def __init__(self):
        super().__init__(temperature=0.1, top_p=0.95)

    def generate(self, prompt, max_new_tokens, alphas):  # pragma: no cover - stub
        return GenerationResult("ack", 1, 1)

    def layers_used(self):  # pragma: no cover - stub
        return []


def _make_research_agent(planner: Planner) -> Agent:
    memory = MemoryStore()
    retriever = MemoryRetriever(memory)
    backend = _StubLanguageBackend()
    safety = SafetyGovernor(
        SafetyConfig(alpha_clip=1.0, toxicity_threshold=1.0, governor_backoff=0.1)
    )
    state = AgentState(
        agent_id="agent-research",
        display_name="agent-research",
        persona_coeffs=PersonaCoeffs(),
        steering_refs=[
            SteeringVectorRef(
                trait="E",
                method="CAA",
                layer_ids=[0],
                vector_store_id="vs-1",
                version="v1",
            )
        ],
        system_prompt="",
        location_id="library",
        goals=["Research"],
        created_at=datetime.now(timezone.utc),
        last_tick=0,
    )
    return Agent(
        run_id="test-run",
        state=state,
        language_backend=backend,
        memory=memory,
        retriever=retriever,
        planner=planner,
        safety_governor=safety,
        reflect_every_n_ticks=1,
    )


def test_agents_emit_research_cycle_despite_advisory_rule():
    planner = Planner()
    agent = _make_research_agent(planner)
    objective = SimpleNamespace(
        objective_id="research-1",
        agent_id="agent-research",
        type="research",
        description="Complete the research cycle",
        requirements={"submit_report": 1},
        progress={"submit_report": 0},
    )
    advisory_rule = Rule(
        rule_id="rule-research",
        text="Keep commerce flowing through the market square.",
        priority="advisory",
        environment_tags=["commerce"],
    )

    observed_actions = []
    for tick in range(4):
        decision = agent.act(
            "Library quiet work.",
            tick,
            current_location="library",
            active_objective=objective,
            recent_dialogue=None,
            rule_context=[advisory_rule],
            peers_present=False,
        )
        observed_actions.append(decision.action_type)

    assert observed_actions == ["research", "research", "cite", "submit_report"]


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
        "steering": {
            "strength": 1.0,
            "coefficients": {"E": 0.1, "A": 0.2, "C": 0.3, "O": 0.0, "N": -0.2},
            "vector_norm": {"E": 1.0, "A": 1.0, "C": 1.0, "O": 1.0, "N": 1.0},
            "metadata_files": {
                "personas": "configs/personas.bigfive.yaml",
                "vectors": "configs/steering.layers.yaml",
            },
        },
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
            "prompt_text": log.prompt_text,
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


@pytest.mark.skipif(bool(MISSING_DEPENDENCIES), reason=MISSING_DEP_REASON)
def test_mock_simulation_reaches_objective(tmp_path):
    from orchestrator.cli import build_agents, build_language_backend, build_objective_manager
    from orchestrator.console_logger import ConsoleLogger
    from orchestrator.runner import SimulationRunner
    from orchestrator.scheduler import Scheduler
    from env.world import World

    config = _build_test_config()
    world = World(data_dir="tests/data")
    world.configure_environment("research", difficulty=1)
    scheduler = Scheduler(world, seed=config["seed"])
    backend = build_language_backend(config, {}, {}, mock=True)
    safety = SafetyGovernor(
        SafetyConfig(alpha_clip=1.0, toxicity_threshold=1.0, governor_backoff=0.1)
    )
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

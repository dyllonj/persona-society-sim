from datetime import datetime

from agents.agent import ActionDecision
from env.world import World
from orchestrator.console_logger import ConsoleLogger
from orchestrator.probes import BehaviorProbeDefinition, LikertProbeDefinition, ProbeManager
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Encounter
from schemas.agent import AgentState, PersonaCoeffs
from storage.log_sink import LogSink


class FakeAgent:
    def __init__(self, agent_id: str, location: str, responses):
        self.state = AgentState(
            agent_id=agent_id,
            display_name=agent_id,
            persona_coeffs=PersonaCoeffs(),
            steering_refs=[],
            role="Probe Responder",
            role_description="Provides canned responses for probe lifecycle tests.",
            system_prompt="Test agent",
            location_id=location,
            goals=["assist neighbors"],
            created_at=datetime.utcnow(),
            last_tick=0,
        )
        self._responses = list(responses)
        self._calls = 0
        self.observations = []

    def act(self, observation: str, tick: int, **_kwargs):
        self.observations.append(observation)
        utterance = self._responses[min(self._calls, len(self._responses) - 1)]
        self._calls += 1
        return ActionDecision(
            action_type="talk",
            params={"utterance": utterance},
            utterance=utterance,
            prompt_text="probe",
            prompt_hash="hash",
            tokens_in=5,
            tokens_out=5,
            steering_snapshot={"C": 0.1},
            layers_used=[],
            safety_event=None,
            plan_metadata={},
            reflection_summary="",
            reflection_implications=[],
        )


class FixedScheduler:
    def __init__(self, world: World, agent_id: str):
        self.world = world
        self.agent_id = agent_id

    def sample(self, _agent_ids, _max_events):
        room = self.world.agent_location(self.agent_id)
        return [Encounter(room_id=room, participants=[self.agent_id], transcript=[])]


class ProbeSpySink(LogSink):
    def __init__(self, run_id: str):
        super().__init__(run_id, db_url=None, parquet_dir=None)
        self.probe_logs = []
        self.behavior_logs = []

    def log_probe(self, log):  # type: ignore[override]
        self.probe_logs.append(log)
        super().log_probe(log)

    def log_behavior_probe(self, log):  # type: ignore[override]
        self.behavior_logs.append(log)
        super().log_behavior_probe(log)


def test_probe_lifecycle_logs_scores():
    world = World(data_dir="tests/data")
    agent = FakeAgent("agent-probe", "library", ["4 - feeling focused.", "I would share supplies." ])
    world.add_agent(agent.state.agent_id, agent.state.location_id)
    scheduler = FixedScheduler(world, agent.state.agent_id)
    sink = ProbeSpySink("probe-run")
    probe_manager = ProbeManager(
        [
            LikertProbeDefinition(
                probe_id="focus",
                question="Rate focus",
                trait="C",
                instructions="Answer 1-5",
                cadence=1,
            )
        ],
        [
            BehaviorProbeDefinition(
                probe_id="share",
                scenario="Share supplies?",
                instructions="Respond SHARE or REFUSE",
                outcomes={"share": ["share"], "refuse": ["refuse"]},
                cadence=1,
            )
        ],
        likert_interval=1,
        behavior_interval=1,
        seed=0,
    )
    runner = SimulationRunner(
        run_id="probe-run",
        world=world,
        scheduler=scheduler,
        agents=[agent],
        log_sink=sink,
        temperature=0.1,
        top_p=0.9,
        console_logger=ConsoleLogger(enabled=False),
        probe_manager=probe_manager,
    )
    runner.run(steps=2, max_events_per_tick=1)

    assert sink.probe_logs, "Likert probes should be logged"
    assert sink.behavior_logs, "Behavior probes should be logged"
    assert any("[Probe]" in obs for obs in agent.observations)
    assert sink.probe_logs[0].score == 4
    assert sink.behavior_logs[0].outcome == "share"

from datetime import datetime, timezone
from types import SimpleNamespace

from agents.agent import Agent
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs, SteeringVectorRef


class StubRetriever:
    def summarize(self, goals, current_tick, focus_terms=None):  # noqa: D401 - simple stub
        return ("stub summary", [SimpleNamespace(memory_id="mem-1")])


class StubPlanner:
    def __init__(self):
        self.calls = 0

    def plan(
        self,
        goals,
        summary,
        current_location=None,
        active_objective=None,
        tick=0,
        rule_context=None,
        observation_hint=None,
    ):
        self.calls += 1
        return PlanSuggestion("talk", {"utterance": "sync"}, "sync")


def _make_agent(reflect_every: int = 5):
    state = AgentState(
        agent_id="agent-1",
        display_name="Agent One",
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
        location_id="town_square",
        goals=["Collaborate"],
        created_at=datetime.now(timezone.utc),
        last_tick=0,
    )
    planner = StubPlanner()
    agent = Agent(
        run_id="run-1",
        state=state,
        language_backend=SimpleNamespace(),
        memory=MemoryStore(),
        retriever=StubRetriever(),
        planner=planner,
        safety_governor=SafetyGovernor(SafetyConfig()),
        reflect_every_n_ticks=reflect_every,
    )
    return agent, planner


def test_agent_plan_cache_expires_after_two_ticks():
    agent, planner = _make_agent()

    first = agent.reflect_and_plan(
        tick=5,
        current_location="market",
        observation="Watching the busy market square.",
        recent_dialogue=None,
    )
    assert planner.calls == 1

    second = agent.reflect_and_plan(
        tick=6,
        current_location="market",
        observation="Still waiting at the market.",
        recent_dialogue=None,
    )
    assert planner.calls == 1
    assert second is first

    third = agent.reflect_and_plan(
        tick=7,
        current_location="market",
        observation="Market crowd is thinning.",
        recent_dialogue=None,
    )
    assert planner.calls == 2
    assert third is not first

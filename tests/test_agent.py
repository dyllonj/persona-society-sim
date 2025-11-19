from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

from agents.agent import Agent
from agents.language_backend import GenerationResult
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion
from metrics.tick_instrumentation import TickInstrumentation
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs, SteeringVectorRef


class StubRetriever:
    def summarize(self, goals, current_tick, focus_terms=None, agent_persona=None):  # noqa: D401 - simple stub
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
        last_reflection_tick=None,
        last_alignment_tick=None,
        observation_keywords=None,
        agent_id=None,
    ):
        self.calls += 1
        return PlanSuggestion("talk", {"utterance": "sync"}, "sync")


class StubLanguageBackend:
    def __init__(self, text: str = "ok"):
        self.text = text

    def generate(self, prompt, max_new_tokens, alphas):  # noqa: D401 - stub
        return GenerationResult(self.text, 10, 12)

    def layers_used(self):  # noqa: D401 - stub
        return []


def _make_agent(
    reflect_every: int = 5,
    *,
    agent_id: str = "agent-1",
    planner: Optional[StubPlanner] = None,
    language_backend: Optional[StubLanguageBackend] = None,
):
    state = AgentState(
        agent_id=agent_id,
        display_name=agent_id,
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
    planner_obj = planner or StubPlanner()
    backend = language_backend or StubLanguageBackend()
    agent = Agent(
        run_id="run-1",
        state=state,
        language_backend=backend,
        memory=MemoryStore(),
        retriever=StubRetriever(),
        planner=planner_obj,
        safety_governor=SafetyGovernor(SafetyConfig()),
        reflect_every_n_ticks=reflect_every,
    )
    return agent, planner_obj


def test_agent_plan_cache_expires_after_two_ticks():
    agent, planner = _make_agent()

    first = agent.reflect_and_plan(
        tick=5,
        current_location="community_center",
        observation="Reviewing notices at the community center.",
        recent_dialogue=None,
    )
    assert planner.calls == 1

    second = agent.reflect_and_plan(
        tick=6,
        current_location="community_center",
        observation="Still waiting near the bulletin board.",
        recent_dialogue=None,
    )
    assert planner.calls == 1
    assert second is first

    third = agent.reflect_and_plan(
        tick=7,
        current_location="community_center",
        observation="Civic crowd is thinning near the hall.",
        recent_dialogue=None,
    )
    assert planner.calls == 2
    assert third is not first


def test_initial_sync_prompts_unique_reduce_duplication():
    class MovePlanner(StubPlanner):
        def plan(
            self,
            goals,
            summary,
            current_location=None,
            active_objective=None,
            tick=0,
            rule_context=None,
            last_reflection_tick=None,
            last_alignment_tick=None,
            observation_keywords=None,
            agent_id=None,
        ):
            self.calls += 1
            return PlanSuggestion(
                "move", {"destination": "community_center"}, "Heading to community center."
            )

    observation = "Location: community_center (Busy). Nearby agents: agent-2."
    agent_one, _ = _make_agent(agent_id="agent-1", planner=MovePlanner())
    agent_two, _ = _make_agent(agent_id="agent-2", planner=MovePlanner())

    decision_one = agent_one.act(
        observation,
        tick=0,
        current_location="community_center",
        recent_dialogue=tuple(),
        rule_context=None,
        peers_present=True,
    )
    decision_two = agent_two.act(
        observation,
        tick=0,
        current_location="community_center",
        recent_dialogue=tuple(),
        rule_context=None,
        peers_present=True,
    )

    assert decision_one.prompt_hash != decision_two.prompt_hash

    instrumentation = TickInstrumentation()
    instrumentation.on_tick_start(0)
    participants = [agent_one.state.agent_id, agent_two.state.agent_id]
    instrumentation.record_action(
        agent_id=agent_one.state.agent_id,
        action_type=decision_one.action_type,
        success=True,
        params=decision_one.params,
        info={},
        steering_snapshot=decision_one.steering_snapshot,
        persona_coeffs=agent_one.state.persona_coeffs.model_dump(),
        encounter_room="community_center",
        encounter_participants=participants,
        satisfaction=0.0,
        prompt_hash=decision_one.prompt_hash,
        prompt_text=decision_one.prompt_text,
        plan_metadata=decision_one.plan_metadata,
    )
    first_share, _ = instrumentation.top_prompt_duplication()

    instrumentation.record_action(
        agent_id=agent_two.state.agent_id,
        action_type=decision_two.action_type,
        success=True,
        params=decision_two.params,
        info={},
        steering_snapshot=decision_two.steering_snapshot,
        persona_coeffs=agent_two.state.persona_coeffs.model_dump(),
        encounter_room="community_center",
        encounter_participants=participants,
        satisfaction=0.0,
        prompt_hash=decision_two.prompt_hash,
        prompt_text=decision_two.prompt_text,
        plan_metadata=decision_two.plan_metadata,
    )
    second_share, _ = instrumentation.top_prompt_duplication()

    assert second_share < first_share

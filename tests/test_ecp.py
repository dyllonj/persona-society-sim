from __future__ import annotations

from datetime import datetime, timezone

from agents.agent import Agent
from agents.language_backend import GenerationResult
from agents.memory import MemoryStore
from agents.planner import PlanSuggestion
from agents.retrieval import MemoryRetriever
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs, SteeringVectorRef
from schemas.memory import MemoryEvent


def test_memory_event_perspective_tags_default_to_legacy_safe_values():
    event = MemoryEvent(
        memory_id="mem-legacy",
        agent_id="agent-a",
        kind="observation",
        tick=1,
        timestamp=datetime.now(timezone.utc),
        text="Legacy observation.",
    )

    assert event.speaker is None
    assert event.self_authored is False


def test_memory_store_persists_perspective_tags():
    store = MemoryStore()

    heard = store.add_event(
        "agent-a",
        "observation",
        1,
        "Meet at the library.",
        0.8,
        speaker="agent-b",
    )
    said = store.add_event(
        "agent-a",
        "observation",
        2,
        "I will cite doc alpha.",
        0.9,
        speaker="agent-a",
    )

    assert heard.speaker == "agent-b"
    assert heard.self_authored is False
    assert said.speaker == "agent-a"
    assert said.self_authored is True


def test_memory_retriever_reprojects_memories_for_agent_perspective():
    store = MemoryStore()
    said = store.add_event(
        "agent-a",
        "observation",
        1,
        "I will cite doc alpha.",
        0.9,
        speaker="agent-a",
    )
    heard = store.add_event(
        "agent-a",
        "observation",
        2,
        "Meet at the library.",
        0.8,
        speaker="agent-b",
    )
    observed = store.add_event(
        "agent-a",
        "observation",
        3,
        "The library is quiet.",
        0.7,
    )
    conflicting = store.add_event(
        "agent-a",
        "observation",
        4,
        "I own this sentence.",
        0.6,
        speaker="agent-b",
        self_authored=True,
    )

    lines = MemoryRetriever(store).reproject_for_agent(
        "agent-a",
        [said, heard, observed, conflicting],
    )

    assert lines == [
        "You said: I will cite doc alpha.",
        "You heard agent-b say: Meet at the library.",
        "You observed: The library is quiet.",
        "You heard agent-b say: I own this sentence.",
    ]


class _SummaryPlanner:
    def __init__(self):
        self.memory_summary = ""

    def plan(self, goals, memory_summary, **kwargs):  # noqa: D401 - test stub
        self.memory_summary = memory_summary
        return PlanSuggestion("talk", {"utterance": "sync"}, "sync")


class _CapturingBackend:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, max_new_tokens, alphas, **kwargs):  # noqa: D401 - test stub
        self.prompt = prompt
        return GenerationResult("ack", 1, 1)

    def layers_used(self):  # noqa: D401 - test stub
        return []


def _make_agent(memory: MemoryStore, planner: _SummaryPlanner, backend: _CapturingBackend) -> Agent:
    state = AgentState(
        agent_id="agent-a",
        display_name="agent-a",
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
        role="ECP Tester",
        role_description="Exercises perspective-safe memory recall.",
        system_prompt="",
        location_id="library",
        goals=["cite library findings"],
        created_at=datetime.now(timezone.utc),
        last_tick=0,
    )
    return Agent(
        run_id="run-ecp",
        state=state,
        language_backend=backend,
        memory=memory,
        retriever=MemoryRetriever(memory),
        planner=planner,
        safety_governor=SafetyGovernor(SafetyConfig()),
        reflect_every_n_ticks=1,
    )


def test_agent_reprojects_recalled_dialogue_in_planning_summary_and_prompt():
    memory = MemoryStore()
    memory.add_event(
        "agent-a",
        "observation",
        1,
        "I will cite doc alpha.",
        1.0,
        speaker="agent-a",
    )
    memory.add_event(
        "agent-a",
        "observation",
        2,
        "Meet at the library.",
        1.0,
        speaker="agent-b",
    )
    planner = _SummaryPlanner()
    backend = _CapturingBackend()
    agent = _make_agent(memory, planner, backend)

    decision = agent.act(
        "Location: library. Quiet work.",
        tick=3,
        current_location="library",
        recent_dialogue=None,
        peers_present=False,
    )

    assert "You said: I will cite doc alpha." in planner.memory_summary
    assert "You heard agent-b say: Meet at the library." in planner.memory_summary
    assert "Recalled dialogue memories:" in decision.prompt_text
    assert "- You said: I will cite doc alpha." in decision.prompt_text
    assert "- You heard agent-b say: Meet at the library." in decision.prompt_text


def test_agent_perceive_tags_speaker_for_ecp():
    """Verify perceive() passes speaker info to memory for ECP projection."""
    memory = MemoryStore()
    planner = _SummaryPlanner()
    backend = _CapturingBackend()
    agent = _make_agent(memory, planner, backend)

    # Simulate ingesting dialogue with speaker tags via act()
    from env.world import RoomUtterance
    dialogue = [
        RoomUtterance(speaker="agent-b", content="Let's meet at the lab.", tick=1),
        RoomUtterance(speaker="agent-a", content="I will bring the data.", tick=1),
    ]

    agent.act(
        "Location: library. Quiet work.",
        tick=2,
        current_location="library",
        recent_dialogue=dialogue,
        peers_present=False,
    )

    # Verify memories have correct speaker tags
    speaker_events = [e for e in memory.events if e.speaker is not None]
    assert len(speaker_events) >= 2
    speakers = {e.speaker for e in speaker_events}
    assert "agent-b" in speakers
    assert "agent-a" in speakers

    # Verify the agent-b event is not self-authored
    agent_b_event = next(e for e in speaker_events if e.speaker == "agent-b")
    assert agent_b_event.self_authored is False

    # Verify the agent-a event is self-authored
    agent_a_event = next(e for e in speaker_events if e.speaker == "agent-a")
    assert agent_a_event.self_authored is True

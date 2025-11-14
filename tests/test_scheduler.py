from __future__ import annotations

from env.world import World
from orchestrator.scheduler import Scheduler


def test_scheduler_includes_recent_room_activity():
    world = World()
    world.add_agent("agent-1", "town_square")
    world.add_agent("agent-2", "town_square")
    world.broadcast("agent-1: hello", room_id="town_square", speaker="agent-1", utterance="hello")
    world.broadcast("agent-1: welcome", room_id="town_square", speaker="agent-1", utterance="welcome")

    scheduler = Scheduler(world, seed=1)
    encounters = scheduler.sample(["agent-1", "agent-2"], max_events=1)

    assert len(encounters) == 1
    encounter = encounters[0]
    assert encounter.room_id == "town_square"
    assert set(encounter.participants) == {"agent-1", "agent-2"}
    assert encounter.transcript
    assert encounter.transcript[-1].speaker == "agent-1"

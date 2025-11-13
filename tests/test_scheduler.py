from __future__ import annotations

from env.world import World
from orchestrator.scheduler import Scheduler


def test_scheduler_includes_recent_room_activity():
    world = World()
    world.add_agent("agent-1", "town_square")
    world.broadcast("agent-1: hello", room_id="town_square")
    world.broadcast("agent-1: welcome", room_id="town_square")

    scheduler = Scheduler(world, seed=1)
    encounters = scheduler.sample(["agent-1"], max_events=1)

    assert len(encounters) == 1
    context = encounters[0].context
    assert "Recent activity here" in context
    assert "agent-1: welcome" in context

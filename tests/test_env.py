from __future__ import annotations

from env.world import World


def test_world_move():
    world = World()
    world.add_agent("agent-1", "town_square")
    world.move_agent("agent-1", "market")
    assert "agent-1" in world.locations["market"].occupants

from __future__ import annotations

from env.world import World


def test_world_move():
    world = World()
    world.add_agent("agent-1", "town_square")
    world.move_agent("agent-1", "market")
    assert "agent-1" in world.locations["market"].occupants


def test_recent_room_context_tracks_messages():
    world = World()
    world.add_agent("agent-1", "town_square")
    world.broadcast("agent-1: hello", room_id="town_square", speaker="agent-1", utterance="hello")
    world.broadcast("agent-1: welcome", room_id="town_square", speaker="agent-1", utterance="welcome")

    context = world.recent_room_context("town_square", limit=2)

    assert "Recent activity here" in context
    assert "agent-1: welcome" in context


def test_recent_room_transcript_includes_metadata():
    world = World()
    world.add_agent("agent-2", "market")
    world.broadcast("agent-2: hi", room_id="market", speaker="agent-2", utterance="hi")

    transcript = world.recent_room_transcript("market", limit=1)

    assert len(transcript) == 1
    assert transcript[0].speaker == "agent-2"
    assert transcript[0].content == "hi"

from __future__ import annotations

from env.world import World
from env import actions


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


def test_fill_field_and_submit_plan_require_unique_entries():
    world = World()
    world.configure_environment("policy", 2)
    world.add_agent("agent-1", "community_center")

    first = actions.fill_field(world, "agent-1", "budget", "Increase grants")
    assert first.success
    duplicate = actions.fill_field(world, "agent-1", "budget", "Duplicate")
    assert not duplicate.success

    not_ready = actions.submit_plan(world, "agent-1")
    assert not not_ready.success

    second = actions.fill_field(world, "agent-1", "staffing", "Hire mentors")
    assert second.success
    ready = actions.submit_plan(world, "agent-1")
    assert ready.success
    assert ready.info["fields_completed"] == "2"


def test_scan_consumes_tokens_once_per_room():
    world = World()
    world.configure_environment("nav", 1)
    world.add_agent("agent-5", "market")

    found = actions.scan(world, "agent-5")
    assert found.success
    assert "token" in found.info

    # Exhaust tokens to force a failure path
    world.location_scan_tokens["market"] = []
    empty = actions.scan(world, "agent-5")
    assert not empty.success

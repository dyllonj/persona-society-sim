from __future__ import annotations

from agents.memory import MemoryStore


def test_memory_store_tracks_events():
    store = MemoryStore()
    store.add_event("a1", "observation", 1, "met b", 0.5)
    store.add_event("a1", "observation", 2, "worked on task", 0.7)
    recent = store.recent_events(limit=1)
    assert len(recent) == 1 and recent[0].tick == 2

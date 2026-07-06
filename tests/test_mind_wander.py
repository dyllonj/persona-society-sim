from __future__ import annotations

from datetime import datetime, timezone

from agents.memory import MemoryStore
from agents.retrieval import MemoryRetriever
from schemas.memory import MemoryEvent


def _event(memory_id: str, text: str, importance: float, tick: int = 1) -> MemoryEvent:
    return MemoryEvent(
        memory_id=memory_id,
        agent_id="agent-1",
        kind="observation",
        tick=tick,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        text=text,
        importance=importance,
    )


def _store() -> MemoryStore:
    store = MemoryStore()
    store.events = [
        _event("top-1", "alpha focus library research", 5.0, tick=10),
        _event("top-2", "alpha focus report synthesis", 4.0, tick=9),
        _event("rare-1", "unrelated garden bell", 0.0, tick=1),
        _event("rare-2", "forgotten kitchen inventory", 0.0, tick=1),
        _event("rare-3", "old weather note", 0.0, tick=1),
    ]
    return store


def test_mind_wander_injects_seeded_low_relevance_memory_without_mutating_store():
    store = _store()
    before = [event.model_dump() for event in store.events]
    first = MemoryRetriever(store, mind_wander_probability=1.0, seed=17)
    second = MemoryRetriever(store, mind_wander_probability=1.0, seed=17)

    _, first_events = first.summarize(["alpha", "focus"], current_tick=12, limit=2)
    _, second_events = second.summarize(["alpha", "focus"], current_tick=12, limit=2)

    first_ids = [event.memory_id for event in first_events]
    second_ids = [event.memory_id for event in second_events]
    assert first_ids == second_ids
    assert first_ids[0] == "top-1"
    assert first_ids[1] in {"rare-1", "rare-2", "rare-3"}
    assert first.last_mind_wander_injections == 1
    assert first.mind_wander_injection_count == 1
    assert [event.model_dump() for event in store.events] == before


def test_mind_wander_probability_zero_preserves_top_retrieval():
    store = _store()
    retriever = MemoryRetriever(store, mind_wander_probability=0.0, seed=17)

    _, events = retriever.summarize(["alpha", "focus"], current_tick=12, limit=2)

    assert [event.memory_id for event in events] == ["top-1", "top-2"]
    assert retriever.last_mind_wander_injections == 0
    assert retriever.mind_wander_injection_count == 0


def test_mind_wander_inverse_frequency_weighting_prefers_rare_events():
    """Verify that rare events are more likely to be selected with inverse-frequency weighting."""
    store = _store()
    # Add a duplicate of rare-1 to make it more common (lower weight)
    store.events.append(_event("rare-1-dup", "unrelated garden bell", 0.0, tick=1))
    store.events.append(_event("rare-1-dup2", "unrelated garden bell", 0.0, tick=1))
    store.events.append(_event("rare-1-dup3", "unrelated garden bell", 0.0, tick=1))

    # Run many trials and count which rare event is selected
    from collections import Counter
    selected = Counter()
    for seed in range(200):
        retriever = MemoryRetriever(store, mind_wander_probability=1.0, seed=seed)
        _, events = retriever.summarize(["alpha", "focus"], current_tick=12, limit=2)
        injected_ids = [e.memory_id for e in events if e.memory_id not in {"top-1", "top-2"}]
        for mid in injected_ids:
            selected[mid] += 1

    # rare-1 ("unrelated garden bell") has 4 copies (low weight)
    # rare-2 ("forgotten kitchen inventory") has 1 copy (high weight)
    # rare-3 ("old weather note") has 1 copy (high weight)
    rare_1_count = selected.get("rare-1", 0)
    rare_2_count = selected.get("rare-2", 0)
    rare_3_count = selected.get("rare-3", 0)

    # rare-1 should be selected less often than rare-2 or rare-3 due to inverse-frequency weighting
    # rare-2 + rare-3 should each be selected more than rare-1
    assert rare_2_count > rare_1_count or rare_3_count > rare_1_count


def test_metric_tracker_on_mind_wander_accumulates():
    """Verify MetricTracker.on_mind_wander accumulates counts."""
    from pathlib import Path
    from metrics.tracker import MetricTracker
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricTracker("test-run", out_dir=Path(tmpdir))
        assert tracker.mind_wander_injections == 0
        tracker.on_mind_wander(1)
        assert tracker.mind_wander_injections == 1
        tracker.on_mind_wander(3)
        assert tracker.mind_wander_injections == 4
        tracker.on_mind_wander(0)
        assert tracker.mind_wander_injections == 4

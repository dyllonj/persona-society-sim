"""Heuristic memory retrieval helpers."""

from __future__ import annotations

from typing import List, Tuple

from schemas.memory import MemoryEvent


class MemoryRetriever:
    def __init__(self, store):
        self.store = store

    def summarize(
        self,
        goals: List[str],
        current_tick: int,
        limit: int = 10,
        focus_terms: List[str] | None = None,
    ) -> Tuple[str, List[MemoryEvent]]:
        query = " ".join(goals) if goals else "daily goings"
        events = self.store.relevant_events(
            query=query,
            current_tick=current_tick,
            limit=limit,
            focus_terms=focus_terms or [],
        )
        if not events:
            return ("No notable memories yet.", [])
        summary = "; ".join(event.text for event in events)
        return summary, events

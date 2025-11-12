"""In-memory observation, reflection, and plan management."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List
from uuid import uuid4

from schemas.memory import MemoryEvent, Plan, Reflection


class MemoryStore:
    def __init__(self):
        self.events: List[MemoryEvent] = []
        self.reflections: List[Reflection] = []
        self.plans: List[Plan] = []

    def add_event(self, agent_id: str, kind: str, tick: int, text: str, importance: float) -> MemoryEvent:
        event = MemoryEvent(
            memory_id=str(uuid4()),
            agent_id=agent_id,
            kind=kind,
            tick=tick,
            timestamp=datetime.utcnow(),
            text=text,
            importance=importance,
        )
        self.events.append(event)
        return event

    def add_reflection(self, agent_id: str, tick: int, text: str, implications: Iterable[str]) -> Reflection:
        reflection = Reflection(
            reflection_id=str(uuid4()),
            agent_id=agent_id,
            tick=tick,
            text=text,
            derived_implications=list(implications),
        )
        self.reflections.append(reflection)
        return reflection

    def add_plan(self, agent_id: str, tick_start: int, tick_end: int, steps: Iterable[str]) -> Plan:
        plan = Plan(
            plan_id=str(uuid4()),
            agent_id=agent_id,
            tick_start=tick_start,
            tick_end=tick_end,
            steps=list(steps),
        )
        self.plans.append(plan)
        return plan

    def recent_events(self, limit: int = 20) -> List[MemoryEvent]:
        return sorted(self.events, key=lambda ev: (ev.tick, ev.timestamp), reverse=True)[:limit]

    def relevant_events(self, query: str, current_tick: int | None = None, limit: int = 5) -> List[MemoryEvent]:
        """Score events by naive keyword overlap × recency × importance."""

        keywords = set(query.lower().split())
        scored = []
        for event in self.events:
            overlap = len(keywords.intersection(event.text.lower().split()))
            recency = 1.0
            if current_tick is not None:
                gap = max(0, current_tick - event.tick)
                recency = max(0.1, 1.0 - 0.01 * gap)
            score = overlap + (event.importance or 0.0) + recency
            scored.append((score, event))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ev for _, ev in scored[:limit]]

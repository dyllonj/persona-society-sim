"""Heuristic memory retrieval helpers."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from schemas.memory import MemoryEvent


class MemoryRetriever:
    def __init__(
        self,
        store,
        mind_wander_probability: float = 0.05,
        seed: int | str | None = 0,
    ):
        self.store = store
        self.mind_wander_probability = self._clamp_probability(mind_wander_probability)
        self._rng = random.Random(seed)
        self.last_mind_wander_injections = 0
        self.mind_wander_injection_count = 0

    def reproject_for_agent(self, agent_id: str, events: Sequence[MemoryEvent]) -> List[str]:
        """Render memories from the acting agent's perspective for prompts."""

        lines: List[str] = []
        for event in events:
            text = " ".join(event.text.split())
            if not text:
                continue
            speaker = event.speaker.strip() if event.speaker else None
            if speaker and speaker != agent_id:
                lines.append(f"You heard {speaker} say: {text}")
            elif event.self_authored or speaker == agent_id:
                lines.append(f"You said: {text}")
            else:
                lines.append(f"You observed: {text}")
        return lines

    def summarize(
        self,
        goals: List[str],
        current_tick: int,
        limit: int = 10,
        focus_terms: List[str] | None = None,
        agent_persona: Dict[str, float] | None = None,
        agent_id: str | None = None,
    ) -> Tuple[str, List[MemoryEvent]]:
        query = " ".join(goals) if goals else "daily goings"
        ranked_events = self.store.ranked_relevant_events(
            query=query,
            current_tick=current_tick,
            focus_terms=focus_terms or [],
            agent_persona=agent_persona,
        )
        events = [event for _, event in ranked_events[:limit]]
        events = self._maybe_inject_mind_wander(events, ranked_events, limit)
        if not events:
            return ("No notable memories yet.", [])
        if agent_id:
            summary = "; ".join(self.reproject_for_agent(agent_id, events))
        else:
            summary = "; ".join(event.text for event in events)
        return summary, events

    def _maybe_inject_mind_wander(
        self,
        events: List[MemoryEvent],
        ranked_events: List[Tuple[float, MemoryEvent]],
        limit: int,
    ) -> List[MemoryEvent]:
        self.last_mind_wander_injections = 0
        if limit <= 0 or self.mind_wander_probability <= 0.0:
            return events
        if self._rng.random() >= self.mind_wander_probability:
            return events

        selected_ids = {event.memory_id for event in events}
        low_relevance_start = max(limit, len(ranked_events) // 2)
        candidate_ranked_events = ranked_events[low_relevance_start:] or ranked_events[limit:]
        candidate_events = [
            event
            for _, event in candidate_ranked_events
            if event.memory_id not in selected_ids
        ]
        if not candidate_events:
            return events

        from collections import Counter
        text_counts = Counter(event.text for event in self.store.events)
        weights = []
        for event in candidate_events:
            count = text_counts.get(event.text, 1)
            weights.append(1.0 / count)
        total_weight = sum(weights)
        if total_weight <= 0:
            injected = self._rng.choice(candidate_events)
        else:
            normalized = [w / total_weight for w in weights]
            injected = self._rng.choices(candidate_events, weights=normalized, k=1)[0]
        updated = list(events)
        if len(updated) >= limit:
            updated[-1] = injected
        else:
            updated.append(injected)
        self.last_mind_wander_injections = 1
        self.mind_wander_injection_count += 1
        return updated

    @staticmethod
    def _clamp_probability(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.05
        return max(0.0, min(1.0, numeric))

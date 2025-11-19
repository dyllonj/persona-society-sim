"""In-memory observation, reflection, and plan management."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence
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
            traits=self._tag_traits(text),
        )
        self.events.append(event)
        return event

    def _tag_traits(self, text: str) -> Dict[str, float]:
        """Heuristic tagging of memory content with Big Five traits."""
        traits = {}
        lowered = text.lower()
        
        # Extraversion: Social interactions
        if any(w in lowered for w in ["party", "social", "meet", "talk", "chat", "community"]):
            traits["Extraversion"] = 0.8
        elif any(w in lowered for w in ["alone", "quiet", "read", "solitary"]):
            traits["Extraversion"] = -0.5
            
        # Conscientiousness: Work and duty
        if any(w in lowered for w in ["work", "task", "plan", "schedule", "duty", "report"]):
            traits["Conscientiousness"] = 0.8
        elif any(w in lowered for w in ["lazy", "late", "forgot", "messy"]):
            traits["Conscientiousness"] = -0.5
            
        # Openness: Learning and novelty
        if any(w in lowered for w in ["learn", "read", "library", "research", "explore", "new"]):
            traits["Openness"] = 0.8
            
        # Agreeableness: Conflict vs Harmony
        if any(w in lowered for w in ["help", "agree", "support", "kind"]):
            traits["Agreeableness"] = 0.6
        elif any(w in lowered for w in ["argue", "fight", "disagree", "rude"]):
            traits["Agreeableness"] = -0.6
            
        # Neuroticism: Stress vs Calm
        if any(w in lowered for w in ["worry", "scared", "stress", "panic", "nervous"]):
            traits["Neuroticism"] = 0.8
        elif any(w in lowered for w in ["calm", "relax", "peace", "steady"]):
            traits["Neuroticism"] = -0.5
            
        return traits

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

    def recent_events(self, limit: int = 30) -> List[MemoryEvent]:
        return sorted(self.events, key=lambda ev: (ev.tick, ev.timestamp), reverse=True)[:limit]

    def relevant_events(
        self,
        query: str,
        current_tick: int | None = None,
        limit: int = 10,
        focus_terms: Optional[Sequence[str]] = None,
        agent_persona: Optional[Dict[str, float]] = None,
    ) -> List[MemoryEvent]:
        """Score events by keyword overlap × recency × importance × trait resonance."""

        keywords = set(query.lower().split())
        focus_tokens: set[str] = set()
        if focus_terms:
            for term in focus_terms:
                focus_tokens.update(term.lower().split())
        scored = []
        for event in self.events:
            tokens = set(event.text.lower().split())
            overlap = len(keywords.intersection(tokens))
            focus_overlap = len(focus_tokens.intersection(tokens))
            focus_bonus = 0.25 * focus_overlap
            recency = 1.0
            if current_tick is not None:
                gap = max(0, current_tick - event.tick)
                recency = max(0.1, 1.0 - 0.01 * gap)
            
            # Trait Resonance: Boost memories that align with the agent's personality
            resonance = 0.0
            if agent_persona and event.traits:
                # Dot product of agent traits and memory traits
                for trait, value in event.traits.items():
                    # Normalize trait key to match persona keys (e.g. "Extraversion" vs "E" or "Extraversion")
                    # Assuming agent_persona uses full names or we map them. 
                    # Let's assume full names for now based on prompt_steering.py, 
                    # but if persona_coeffs uses "E", "A", etc., we need mapping.
                    # The schema uses "E", "A", "C", "O", "N".
                    short_map = {
                        "Extraversion": "E",
                        "Agreeableness": "A",
                        "Conscientiousness": "C",
                        "Neuroticism": "N",
                        "Openness": "O"
                    }
                    pkey = short_map.get(trait, trait)
                    agent_val = agent_persona.get(pkey, 0.0)
                    resonance += value * agent_val
            
            score = overlap + focus_bonus + (event.importance or 0.0) + recency + (resonance * 0.5)
            # Deterministic jitter breaks ties so neighboring agents don't grab identical bundles.
            jitter_seed = hashlib.sha256(event.memory_id.encode("utf-8")).digest()
            jitter = int.from_bytes(jitter_seed[:2], "big") / 65535.0  # 0-1 range
            score += jitter * 0.05
            scored.append((score, event))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ev for _, ev in scored[:limit]]

"""Sampling encounters among agents each tick."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from env.world import World


@dataclass
class Encounter:
    agent_id: str
    context: str


class Scheduler:
    def __init__(self, world: World, seed: int = 7):
        self.world = world
        self.random = random.Random(seed)

    def sample(self, agent_ids: Sequence[str], max_events: int) -> List[Encounter]:
        picks = self.random.sample(agent_ids, k=min(max_events, len(agent_ids)))
        encounters = []
        for agent_id in picks:
            context = self.world.sample_context(agent_id)
            encounters.append(Encounter(agent_id=agent_id, context=context))
        return encounters

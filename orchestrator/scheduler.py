"""Sampling group encounters among agents each tick."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from env.world import RoomUtterance, World


@dataclass
class Encounter:
    room_id: str
    participants: List[str]
    transcript: List[RoomUtterance]


class Scheduler:
    def __init__(self, world: World, seed: int = 7):
        self.world = world
        self.random = random.Random(seed)

    def _pick_group_size(self, remaining: int) -> int:
        if remaining >= 3:
            return self.random.choice((2, 3))
        return remaining

    def sample(self, agent_ids: Sequence[str], max_events: int) -> List[Encounter]:
        available = set(agent_ids)
        encounters: List[Encounter] = []
        room_items = list(self.world.locations.items())
        self.random.shuffle(room_items)

        for room_id, location in room_items:
            if len(encounters) >= max_events:
                break
            room_agents = [agent for agent in location.occupants if agent in available]
            if not room_agents:
                continue
            self.random.shuffle(room_agents)

            while room_agents and len(encounters) < max_events:
                group_size = self._pick_group_size(len(room_agents))
                group = room_agents[:group_size]
                room_agents = room_agents[group_size:]
                for agent in group:
                    available.discard(agent)
                transcript = self.world.recent_room_transcript(room_id, limit=5)
                encounters.append(
                    Encounter(
                        room_id=room_id,
                        participants=group,
                        transcript=transcript,
                    )
                )

        return encounters

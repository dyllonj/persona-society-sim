"""Text-only town environment state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Location:
    name: str
    description: str
    occupants: Set[str] = field(default_factory=set)


class World:
    def __init__(self):
        self.locations: Dict[str, Location] = {
            "town_square": Location("town_square", "Central gathering spot"),
            "community_center": Location("community_center", "Meetings, classes, and civic events"),
            "market": Location("market", "Barter goods and post offers"),
            "library": Location("library", "Quiet work and study"),
        }
        self.noticeboard: List[str] = []
        self.tick = 0

    def add_agent(self, agent_id: str, location_id: str) -> None:
        location = self.locations.setdefault(location_id, Location(location_id, ""))
        location.occupants.add(agent_id)

    def move_agent(self, agent_id: str, destination: str) -> None:
        for location in self.locations.values():
            location.occupants.discard(agent_id)
        self.locations.setdefault(destination, Location(destination, "")).occupants.add(agent_id)

    def broadcast(self, msg: str) -> None:
        self.noticeboard.append(msg)

    def sample_context(self, agent_id: str) -> str:
        for location in self.locations.values():
            if agent_id in location.occupants:
                peers = location.occupants - {agent_id}
                peer_list = ", ".join(sorted(peers)) or "no one else"
                return f"You are at {location.name} with {peer_list}."
        return "You are wandering the outskirts alone."

    def step(self) -> None:
        self.tick += 1

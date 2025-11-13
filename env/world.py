"""Text-only town environment state container."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set


@dataclass
class Location:
    name: str
    description: str
    occupants: Set[str] = field(default_factory=set)


class World:
    def __init__(self, room_history_limit: int = 8):
        self.locations: Dict[str, Location] = {
            "town_square": Location("town_square", "Central gathering spot"),
            "community_center": Location("community_center", "Meetings, classes, and civic events"),
            "market": Location("market", "Barter goods and post offers"),
            "library": Location("library", "Quiet work and study"),
        }
        self.noticeboard: List[str] = []
        self.room_history: Dict[str, Deque[str]] = {}
        self._room_history_limit = room_history_limit
        self.tick = 0

    def add_agent(self, agent_id: str, location_id: str) -> None:
        location = self.locations.setdefault(location_id, Location(location_id, ""))
        location.occupants.add(agent_id)

    def move_agent(self, agent_id: str, destination: str) -> None:
        for location in self.locations.values():
            location.occupants.discard(agent_id)
        self.locations.setdefault(destination, Location(destination, "")).occupants.add(agent_id)

    def broadcast(self, msg: str, room_id: str | None = None) -> None:
        self.noticeboard.append(msg)
        if not room_id:
            return
        history = self.room_history.setdefault(
            room_id, deque(maxlen=self._room_history_limit)
        )
        history.append(msg)

    def recent_room_context(self, room_id: str, limit: int = 3) -> str:
        """Return a formatted snippet of recent messages for ``room_id``."""

        history = self.room_history.get(room_id)
        if not history:
            return ""

        tail = list(history)[-limit:]
        formatted = "\n".join(f"- {entry}" for entry in tail)
        return f"Recent activity here:\n{formatted}"

    def sample_context(self, agent_id: str) -> str:
        for location in self.locations.values():
            if agent_id in location.occupants:
                peers = location.occupants - {agent_id}
                peer_list = ", ".join(sorted(peers)) or "no one else"
                return f"You are at {location.name} with {peer_list}."
        return "You are wandering the outskirts alone."

    def agent_location(self, agent_id: str) -> str:
        for location in self.locations.values():
            if agent_id in location.occupants:
                return location.name
        return "unknown"

    def step(self) -> None:
        self.tick += 1

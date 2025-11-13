"""Text-only town environment state container."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set, Optional
from pathlib import Path
import json


@dataclass
class Location:
    name: str
    description: str
    occupants: Set[str] = field(default_factory=set)


class World:
    def __init__(self, room_history_limit: int = 8, data_dir: str = "data"):
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
        self.data_dir = Path(data_dir)

        # Research Sprint corpus and per-agent scratchpads
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.targets: Dict[str, Dict[str, str]] = {}
        self.agent_research: Dict[str, Dict[str, object]] = {}
        self._load_corpus()

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

    # ---- Research corpus helpers ----

    def _load_corpus(self) -> None:
        targets_path = self.data_dir / "corpus" / "targets.json"
        if not targets_path.exists():
            return
        with targets_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # data format: {"docs": {doc_id: {fact_id: answer}}, "targets": {fact_id: {"doc_id": str, "answer": str}}}
        self.corpus = data.get("docs", {})
        self.targets = data.get("targets", {})

    def _ensure_agent_research(self, agent_id: str) -> Dict[str, object]:
        return self.agent_research.setdefault(
            agent_id,
            {"accessed_docs": set(), "found_facts": {}, "citations": set()},
        )

    def research_access(self, agent_id: str, doc_id: Optional[str] = None, query: Optional[str] = None) -> Dict[str, object]:
        state = self._ensure_agent_research(agent_id)
        info: Dict[str, object] = {"doc_id": None, "facts_found": []}
        # Resolve doc by id or naive query match (fact id or doc id substring)
        chosen_doc: Optional[str] = None
        if doc_id and doc_id in self.corpus:
            chosen_doc = doc_id
        elif query:
            # If query equals a fact id, pick its doc
            if query in self.targets:
                chosen_doc = self.targets[query]["doc_id"]
            else:
                for d in self.corpus:
                    if query.lower() in d.lower():
                        chosen_doc = d
                        break
        if not chosen_doc and self.corpus:
            chosen_doc = next(iter(self.corpus.keys()))
        if not chosen_doc:
            return info
        state["accessed_docs"].add(chosen_doc)
        info["doc_id"] = chosen_doc
        # Reveal facts in chosen doc that are also targets
        facts = self.corpus.get(chosen_doc, {})
        for fact_id, answer in facts.items():
            if fact_id in self.targets:
                state["found_facts"][fact_id] = answer
                info["facts_found"].append({"fact_id": fact_id, "answer": answer})
        return info

    def add_citation(self, agent_id: str, doc_id: str) -> None:
        state = self._ensure_agent_research(agent_id)
        state["citations"].add(doc_id)

    def grade_report(self, agent_id: str) -> Dict[str, object]:
        """Compare found facts + citations against targets and compute a simple score."""
        state = self._ensure_agent_research(agent_id)
        found: Dict[str, str] = state.get("found_facts", {})  # type: ignore
        citations = set(state.get("citations", set()))  # type: ignore

        total = len(self.targets)
        correct = sum(1 for k, v in self.targets.items() if found.get(k) == v.get("answer"))
        # Valid citations are those where the cited doc contains at least one target fact found
        valid_cites = 0
        for c in citations:
            facts = self.corpus.get(c, {})
            if any(fid in self.targets for fid in facts.keys()):
                valid_cites += 1
        reward_points = correct * 1.0 + valid_cites * 0.5
        return {
            "targets_total": total,
            "facts_correct": correct,
            "citations_valid": valid_cites,
            "reward_points": reward_points,
        }

    def recent_room_context(self, room_id: str, limit: int = 3) -> str:
        """Return a formatted snippet of recent messages for ``room_id``."""

        history = self.room_history.get(room_id)
        if not history:
            return ""

        tail = list(history)[-limit:]
        formatted = "\n".join(f"- {entry}" for entry in tail)
        return f"Recent activity here:\n{formatted}"

    def sample_context(self, agent_id: str) -> str:
        """Provide a neutral scene description without second-person voice.

        This avoids POV drift like leading with "You:" and reduces
        contradictions when move actions immediately follow planning.
        """
        for location in self.locations.values():
            if agent_id in location.occupants:
                peers = location.occupants - {agent_id}
                peer_list = ", ".join(sorted(peers)) or "no other agents"
                return (
                    f"Location: {location.name}. Nearby agents: {peer_list}."
                )
        return "Location: outskirts. Nearby agents: none."

    def agent_location(self, agent_id: str) -> str:
        for location in self.locations.values():
            if agent_id in location.occupants:
                return location.name
        return "unknown"

    def step(self) -> None:
        self.tick += 1

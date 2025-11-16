"""Text-only town environment state container."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set
from pathlib import Path
import json

from env.economy import Economy
from env.institutions import InstitutionManager
from schemas.agent import Rule


@dataclass
class Location:
    name: str
    description: str
    occupants: Set[str] = field(default_factory=set)
    x: float = 0.0
    y: float = 0.0
    neighbors: Set[str] = field(default_factory=set)
    capacity: int = 8
    resources: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "coords": {"x": self.x, "y": self.y},
            "neighbors": sorted(self.neighbors),
            "capacity": self.capacity,
            "resources": dict(self.resources),
            "occupants": sorted(self.occupants),
        }


@dataclass
class RoomUtterance:
    speaker: str
    content: str
    tick: int


class World:
    def __init__(self, room_history_limit: int = 8, data_dir: str = "data"):
        self.locations: Dict[str, Location] = {
            "town_square": Location(
                "town_square",
                "Central gathering spot",
                x=0.0,
                y=0.0,
                neighbors={"market", "community_center", "library"},
                capacity=20,
                resources={"credits": 20, "tokens": 3},
            ),
            "community_center": Location(
                "community_center",
                "Meetings, classes, and civic events",
                x=-6.0,
                y=-8.0,
                neighbors={"town_square", "market", "library"},
                capacity=14,
                resources={"supplies": 5, "credits": 15},
            ),
            "market": Location(
                "market",
                "Barter goods and post offers",
                x=10.0,
                y=-2.0,
                neighbors={"town_square", "community_center"},
                capacity=16,
                resources={"produce": 12, "tools": 6, "credits": 50},
            ),
            "library": Location(
                "library",
                "Quiet work and study",
                x=-9.0,
                y=7.5,
                neighbors={"town_square", "community_center"},
                capacity=10,
                resources={"scrolls": 4, "credits": 5},
            ),
        }
        self.noticeboard: List[str] = []
        self.room_history: Dict[str, Deque[RoomUtterance]] = {}
        self._room_history_limit = room_history_limit
        self.tick = 0
        self.data_dir = Path(data_dir)

        # Lightweight economy + policy state
        self.economy = Economy()
        self.institutions = InstitutionManager()
        self.agent_checklists: Dict[str, Dict[str, str]] = {}
        self.agent_policy_plans: Dict[str, Dict[str, str]] = {}
        self.agent_scan_tokens: Dict[str, Set[str]] = {}
        self.location_scan_tokens: Dict[str, List[str]] = {}
        self.policy_required_fields = 3
        self.nav_token_goal = 3
        self.research_fact_goal = 3
        self.environment = "research"

        # Research Sprint corpus and per-agent scratchpads
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.targets: Dict[str, Dict[str, str]] = {}
        self.agent_research: Dict[str, Dict[str, object]] = {}
        self._load_corpus()
        self._seed_scan_tokens()
        self._seed_rules()

    def configure_environment(self, env_name: str, difficulty: int) -> None:
        self.environment = env_name
        if env_name == "policy":
            self.policy_required_fields = max(1, difficulty)
        elif env_name == "nav":
            self.nav_token_goal = max(1, difficulty)
        elif env_name == "research":
            self.research_fact_goal = max(1, difficulty)
        self._seed_scan_tokens()

    def add_agent(self, agent_id: str, location_id: str) -> None:
        location = self.locations.setdefault(location_id, Location(location_id, ""))
        location.occupants.add(agent_id)

    def move_agent(self, agent_id: str, destination: str) -> None:
        for location in self.locations.values():
            location.occupants.discard(agent_id)
        self.locations.setdefault(destination, Location(destination, "")).occupants.add(agent_id)

    # ---- economy + inventory helpers ----

    def adjust_resource(self, agent_id: str, item: str, delta: int) -> int:
        return self.economy.adjust(agent_id, item, delta)

    def resource_balance(self, agent_id: str, item: str) -> int:
        return self.economy.balance(agent_id, item)

    def agent_inventory(self, agent_id: str) -> Dict[str, int]:
        return self.economy.snapshot().get(agent_id, {})

    def trade_with_location(
        self,
        agent_id: str,
        location_id: str,
        item: str,
        qty: int,
        price: float,
        side: str,
    ) -> tuple[bool, Dict[str, str]]:
        location = self.locations.get(location_id)
        if not location:
            return False, {"error": "invalid_location"}
        if qty <= 0:
            return False, {"error": "invalid_qty"}
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell"}:
            normalized_side = "buy"
        total_price = int(max(0, round(price * qty)))
        if normalized_side == "buy":
            stock = location.resources.get(item, 0)
            if stock < qty:
                return False, {"error": "insufficient_stock"}
            if self.resource_balance(agent_id, "credits") < total_price:
                return False, {"error": "insufficient_credits"}
            location.resources[item] = stock - qty
            self.adjust_resource(agent_id, item, qty)
            if total_price:
                location.resources["credits"] = location.resources.get("credits", 0) + total_price
                self.adjust_resource(agent_id, "credits", -total_price)
            return True, {
                "note": "purchased",
                "price_paid": str(total_price),
                "balance": str(self.resource_balance(agent_id, item)),
            }
        # selling path
        if self.resource_balance(agent_id, item) < qty:
            return False, {"error": "insufficient_inventory"}
        if location.resources.get("credits", 0) < total_price:
            return False, {"error": "location_insufficient_credits"}
        location.resources[item] = location.resources.get(item, 0) + qty
        self.adjust_resource(agent_id, item, -qty)
        if total_price:
            location.resources["credits"] = max(0, location.resources.get("credits", 0) - total_price)
            self.adjust_resource(agent_id, "credits", total_price)
        return True, {
            "note": "sold",
            "price_paid": str(total_price),
            "balance": str(self.resource_balance(agent_id, item)),
        }

    def serialize(self, include_agents: bool = False, agent_ids: Optional[Iterable[str]] = None) -> Dict[str, object]:
        state: Dict[str, object] = {
            "tick": self.tick,
            "locations": {loc_id: loc.to_dict() for loc_id, loc in self.locations.items()},
            "rules": [rule.model_dump() for rule in self.institutions.active_rules()],
        }
        if include_agents:
            ids = list(agent_ids) if agent_ids else sorted(
                {agent for loc in self.locations.values() for agent in loc.occupants}
            )
            holdings_snapshot = self.economy.snapshot()
            state["agents"] = {
                agent_id: {
                    "location_id": self.agent_location(agent_id),
                    "inventory": holdings_snapshot.get(agent_id, {}),
                }
                for agent_id in ids
            }
        return state

    def enact_plan_rule(self, agent_id: str) -> Optional[Rule]:
        plan = self.agent_policy_plans.get(agent_id)
        if not plan:
            return None
        summary = plan.get("summary") or f"Plan submitted by {agent_id}"
        rule = self.institutions.propose_rule(agent_id, summary, self.tick)
        return self.institutions.enact_rule(rule.rule_id, self.tick)

    def institutional_guidance(self) -> List[Rule]:
        """Return active rules, marking advisory overrides for current environment."""

        guidance: List[Rule] = []
        for rule in self.institutions.active_rules():
            normalized_tags = {tag.lower() for tag in rule.environment_tags}
            is_commerce_rule = (
                "commerce" in normalized_tags
                or "market" in normalized_tags
                or "economy" in normalized_tags
                or "keep commerce" in rule.text.lower()
            )
            if self.environment == "research" and is_commerce_rule:
                guidance.append(rule.model_copy(update={"priority": "advisory"}))
            else:
                guidance.append(rule)
        return guidance

    # ---- checklist helpers ----

    def record_checklist_field(self, agent_id: str, field_name: str, value: str) -> bool:
        checklist = self.agent_checklists.setdefault(agent_id, {})
        if field_name in checklist:
            return False
        checklist[field_name] = value
        return True

    def checklist_fields_completed(self, agent_id: str) -> int:
        return len(self.agent_checklists.get(agent_id, {}))

    def policy_plan_ready(self, agent_id: str) -> bool:
        return self.checklist_fields_completed(agent_id) >= self.policy_required_fields

    # ---- scan token helpers ----

    def _seed_scan_tokens(self) -> None:
        tokens_per_location = max(3, self.nav_token_goal * 2)
        self.location_scan_tokens = {
            loc_id: [f"{loc_id}_token_{i}" for i in range(1, tokens_per_location + 1)]
            for loc_id in self.locations
        }

    def acquire_scan_token(self, agent_id: str, location_id: str) -> str | None:
        tokens = self.location_scan_tokens.setdefault(location_id, [])
        if not tokens:
            return None
        token = tokens.pop(0)
        owned = self.agent_scan_tokens.setdefault(agent_id, set())
        owned.add(token)
        return token

    def broadcast(
        self,
        msg: str,
        room_id: str | None = None,
        *,
        speaker: Optional[str] = None,
        utterance: Optional[str] = None,
    ) -> None:
        self.noticeboard.append(msg)
        if not room_id:
            return
        history = self.room_history.setdefault(
            room_id, deque(maxlen=self._room_history_limit)
        )
        history.append(
            RoomUtterance(
                speaker=speaker or "system",
                content=utterance or msg,
                tick=self.tick,
            )
        )

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
                target_answer = self.targets[fact_id]["answer"]
                is_correct = answer == target_answer
                state["found_facts"][fact_id] = answer
                info["facts_found"].append(
                    {
                        "fact_id": fact_id,
                        "answer": answer,
                        "target_answer": target_answer,
                        "correct": is_correct,
                    }
                )
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

    def recent_room_transcript(self, room_id: str, limit: int = 3) -> List[RoomUtterance]:
        """Return the most recent structured utterances for ``room_id``."""

        history = self.room_history.get(room_id)
        if not history:
            return []
        tail = list(history)[-limit:]
        return tail

    def recent_room_context(self, room_id: str, limit: int = 3) -> str:
        """Return a formatted snippet of recent messages for ``room_id``."""

        transcript = self.recent_room_transcript(room_id, limit=limit)
        if not transcript:
            return ""

        formatted = "\n".join(
            f"- {entry.speaker}: {entry.content}" for entry in transcript
        )
        return f"Recent activity here:\n{formatted}"

    def sample_context(self, agent_id: str) -> str:
        """Provide a neutral scene description including physical metadata."""

        for location in self.locations.values():
            if agent_id in location.occupants:
                peers = location.occupants - {agent_id}
                peer_list = ", ".join(sorted(peers)) or "no other agents"
                resources = ", ".join(f"{k}:{v}" for k, v in location.resources.items()) or "no supplies"
                neighbors = ", ".join(sorted(location.neighbors)) or "isolated"
                occupancy = f"{len(location.occupants)}/{location.capacity}"
                return (
                    f"Location: {location.name} ({location.description}). "
                    f"Capacity {occupancy}. Nearby agents: {peer_list}. "
                    f"Neighbors: {neighbors}. Resources: {resources}."
                )
        return "Location: outskirts. Nearby agents: none."

    def agent_location(self, agent_id: str) -> str:
        for location in self.locations.values():
            if agent_id in location.occupants:
                return location.name
        return "unknown"

    def step(self) -> None:
        self.tick += 1

    def _seed_rules(self) -> None:
        """Provide a baseline civic rule so planners have initial guidance."""

        if self.institutions.active_rules():
            return
        priority = "advisory" if self.environment == "research" else "mandatory"
        baseline = self.institutions.propose_rule(
            "council",
            "Keep commerce flowing through the market square.",
            self.tick,
            priority=priority,
            environment_tags=["commerce", "market"],
        )
        self.institutions.enact_rule(baseline.rule_id, self.tick)

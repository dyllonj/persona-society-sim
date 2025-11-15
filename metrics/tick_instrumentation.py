"""Tick-level instrumentation for graph + macro metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from metrics.persona_bands import band_metadata, trait_band_key
from schemas.logs import Edge


@dataclass
class GraphInput:
    trait_key: Optional[str]
    edges: List[Edge]
    band_metadata: Dict[str, object]


@dataclass
class MacroInput:
    trait_key: Optional[str]
    cooperation_events: List[str]
    conflicts: int
    enforcement_cost: float
    band_metadata: Dict[str, object]
    wealth: Dict[str, float]
    opinions: Dict[str, float]


class TickInstrumentation:
    COOP_ACTIONS = {"talk", "trade", "work", "gift", "research", "cite", "submit_report"}
    CONFLICT_ACTIONS = {"steal", "report"}

    def __init__(self) -> None:
        self._edges: Dict[Optional[str], List[Edge]] = defaultdict(list)
        self._cooperation_events: Dict[Optional[str], List[str]] = defaultdict(list)
        self._conflicts: Dict[Optional[str], int] = defaultdict(int)
        self._enforcement_cost: Dict[Optional[str], float] = defaultdict(float)
        self._band_members: Dict[Optional[str], Set[str]] = defaultdict(set)
        self._agent_opinions: Dict[str, float] = {}

    def on_tick_start(self, tick: int) -> None:
        self._edges.clear()
        self._cooperation_events.clear()
        self._conflicts.clear()
        self._enforcement_cost.clear()
        self._band_members.clear()
        self._agent_opinions.clear()

    def record_action(
        self,
        *,
        agent_id: str,
        action_type: str,
        success: bool,
        params: Dict[str, str],
        info: Dict[str, Any],
        steering_snapshot: Dict[str, float],
        persona_coeffs: Dict[str, float],
        encounter_room: str,
        encounter_participants: Iterable[str],
        satisfaction: float,
    ) -> None:
        trait_key = trait_band_key(persona_coeffs, steering_snapshot)
        self._band_members[None].add(agent_id)
        if trait_key:
            self._band_members[trait_key].add(agent_id)
        self._agent_opinions[agent_id] = satisfaction

        if action_type == "talk":
            self._record_message_edges(agent_id, encounter_participants, trait_key)
        elif action_type in {"trade", "gift"}:
            self._record_trade_edge(agent_id, params, trait_key, encounter_room)
        elif action_type == "enforce":
            self._record_enforcement(agent_id, params, info, trait_key)

        if success and action_type in self.COOP_ACTIONS:
            self._record_cooperation(action_type, agent_id, trait_key)
        if success and action_type in self.CONFLICT_ACTIONS:
            self._increment_conflict(trait_key)
        if action_type == "enforce":
            self._increment_enforcement_cost(info, trait_key)

    # ---- aggregation ----

    def graph_inputs(self) -> List[GraphInput]:
        inputs: List[GraphInput] = []
        for trait_key, edges in self._edges.items():
            if not edges:
                continue
            inputs.append(GraphInput(trait_key, list(edges), band_metadata(trait_key)))
        return inputs

    def macro_inputs(
        self,
        wealth_snapshot: Dict[str, Dict[str, int]],
        opinions: Dict[str, float],
    ) -> List[MacroInput]:
        wealth_scalar = {agent: float(sum(holdings.values())) for agent, holdings in wealth_snapshot.items()}
        combined_members: Dict[Optional[str], Set[str]] = defaultdict(set)
        for trait_key, members in self._band_members.items():
            combined_members[trait_key] = set(members)
        if None not in combined_members:
            combined_members[None] = set(wealth_scalar.keys()) or set(self._agent_opinions.keys())

        inputs: List[MacroInput] = []
        for trait_key, members in combined_members.items():
            if not members:
                continue
            wealth_slice = {agent: wealth_scalar.get(agent, 0.0) for agent in members}
            opinion_slice = {
                agent: opinions.get(agent, self._agent_opinions.get(agent, 0.0)) for agent in members
            }
            inputs.append(
                MacroInput(
                    trait_key=trait_key,
                    cooperation_events=list(self._cooperation_events.get(trait_key, [])),
                    conflicts=self._conflicts.get(trait_key, 0),
                    enforcement_cost=self._enforcement_cost.get(trait_key, 0.0),
                    band_metadata=band_metadata(trait_key),
                    wealth=wealth_slice,
                    opinions=opinion_slice,
                )
            )
        if not inputs:
            inputs.append(
                MacroInput(
                    trait_key=None,
                    cooperation_events=[],
                    conflicts=0,
                    enforcement_cost=0.0,
                    band_metadata={},
                    wealth={agent: float(sum(holdings.values())) for agent, holdings in wealth_snapshot.items()},
                    opinions=dict(opinions),
                )
            )
        return inputs

    # ---- internals ----

    def _record_message_edges(
        self, agent_id: str, participants: Iterable[str], trait_key: Optional[str]
    ) -> None:
        peers = [peer for peer in participants if peer != agent_id]
        for peer in peers:
            edge = Edge(src=agent_id, dst=peer, weight=1.0, kind="message")
            self._append_edge(edge, trait_key)

    def _record_trade_edge(
        self, agent_id: str, params: Dict[str, str], trait_key: Optional[str], room_id: str
    ) -> None:
        target = params.get("recipient") or params.get("counterparty") or room_id
        edge = Edge(src=agent_id, dst=target, weight=1.0, kind="trade")
        self._append_edge(edge, trait_key)

    def _record_enforcement(
        self, agent_id: str, params: Dict[str, str], info: Dict[str, Any], trait_key: Optional[str]
    ) -> None:
        target = params.get("target_id") or info.get("target_id") or params.get("recipient")
        if not target:
            return
        edge = Edge(src=agent_id, dst=target, weight=1.0, kind="sanction")
        self._append_edge(edge, trait_key)

    def _record_cooperation(self, action_type: str, agent_id: str, trait_key: Optional[str]) -> None:
        event = f"{action_type}:{agent_id}"
        self._cooperation_events[None].append(event)
        if trait_key:
            self._cooperation_events[trait_key].append(event)

    def _increment_conflict(self, trait_key: Optional[str]) -> None:
        self._conflicts[None] += 1
        if trait_key:
            self._conflicts[trait_key] += 1

    def _increment_enforcement_cost(self, info: Dict[str, Any], trait_key: Optional[str]) -> None:
        cost_raw = info.get("cost") or info.get("amount") or "1"
        try:
            cost = float(cost_raw)
        except (TypeError, ValueError):
            cost = 1.0
        self._enforcement_cost[None] += cost
        if trait_key:
            self._enforcement_cost[trait_key] += cost

    def _append_edge(self, edge: Edge, trait_key: Optional[str]) -> None:
        self._edges[None].append(edge)
        if trait_key:
            self._edges[trait_key].append(edge)

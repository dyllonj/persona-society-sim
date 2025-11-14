"""Lightweight barter economy bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Economy:
    """Track per-agent holdings and simple trade execution."""

    holdings: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def adjust(self, agent_id: str, item: str, delta: int) -> int:
        inventory = self.holdings.setdefault(agent_id, {})
        new_balance = max(0, inventory.get(item, 0) + delta)
        inventory[item] = new_balance
        return new_balance

    def balance(self, agent_id: str, item: str) -> int:
        return self.holdings.get(agent_id, {}).get(item, 0)

    def transfer(self, giver: str, receiver: str, item: str, qty: int) -> bool:
        if self.balance(giver, item) < qty:
            return False
        self.adjust(giver, item, -qty)
        self.adjust(receiver, item, qty)
        return True

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {agent: holdings.copy() for agent, holdings in self.holdings.items()}

    def apply_trade(
        self,
        *,
        buyer: str,
        seller: str,
        item: str,
        qty: int,
        unit_price: float,
    ) -> bool:
        """Execute a trade, exchanging credits for an item quantity."""

        total_price = int(max(0, round(unit_price * qty)))
        if self.balance(seller, item) < qty:
            return False
        if self.balance(buyer, "credits") < total_price:
            return False
        self.adjust(seller, item, -qty)
        self.adjust(buyer, item, qty)
        if total_price:
            self.adjust(buyer, "credits", -total_price)
            self.adjust(seller, "credits", total_price)
        return True

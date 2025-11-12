"""Lightweight barter economy bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Holding:
    item: str
    qty: int


@dataclass
class Economy:
    balances: Dict[str, Dict[str, Holding]] = field(default_factory=dict)

    def apply_trade(self, buyer: str, seller: str, item: str, qty: int, price: float) -> None:
        buyer_holdings = self.balances.setdefault(buyer, {})
        seller_holdings = self.balances.setdefault(seller, {})
        buyer_holdings[item] = Holding(item, buyer_holdings.get(item, Holding(item, 0)).qty + qty)
        seller_holdings[item] = Holding(item, max(0, seller_holdings.get(item, Holding(item, 0)).qty - qty))
        # TODO: integrate currency ledger; placeholder ensures structure exists.

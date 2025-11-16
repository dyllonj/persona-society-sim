"""Simple rule proposal and enforcement helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from uuid import uuid4

from schemas.agent import Rule


class InstitutionManager:
    def __init__(self):
        self.rules: Dict[str, Rule] = {}

    def propose_rule(
        self,
        proposer_id: str,
        text: str,
        tick: int,
        *,
        priority: str = "mandatory",
        environment_tags: Optional[Sequence[str]] = None,
    ) -> Rule:
        rule = Rule(
            rule_id=str(uuid4()),
            text=text,
            proposer_id=proposer_id,
            enacted_at_tick=tick,
            active=False,
            priority="advisory" if priority == "advisory" else "mandatory",
            environment_tags=list(environment_tags or []),
        )
        self.rules[rule.rule_id] = rule
        return rule

    def enact_rule(self, rule_id: str, tick: int) -> Rule:
        rule = self.rules[rule_id]
        rule.active = True
        rule.enacted_at_tick = tick
        return rule

    def active_rules(self) -> List[Rule]:
        return [rule for rule in self.rules.values() if rule.active]

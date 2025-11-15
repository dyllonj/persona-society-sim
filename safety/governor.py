"""Lightweight safety governor for textual generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from schemas.logs import SafetyEvent


DEFAULT_BANNED_PHRASES = [
    "hate speech",
    "bomb recipe",
    "harm yourself",
    "kill",
    "attack plan",
]


@dataclass
class SafetyConfig:
    alpha_clip: float = 1.0
    toxicity_threshold: float = 0.4  # placeholder for future classifier
    governor_backoff: float = 0.2
    banned_phrases: Iterable[str] = tuple(DEFAULT_BANNED_PHRASES)


class SafetyGovernor:
    """Detect naive safety triggers and dampen persona coefficients when needed."""

    def __init__(self, config: SafetyConfig):
        self.config = config

    def clamp(self, value: float) -> float:
        return max(-self.config.alpha_clip, min(self.config.alpha_clip, value))

    def evaluate(
        self,
        run_id: str,
        agent_id: str,
        text: str,
        tick: int,
        current_alphas: Dict[str, float],
    ) -> Optional[SafetyEvent]:
        lowered = text.lower()
        hits: List[str] = [phrase for phrase in self.config.banned_phrases if phrase in lowered]
        if not hits:
            return None
        severity = "high" if len(hits) > 1 else "medium"
        alpha_delta = {
            trait: self.clamp(-self.config.governor_backoff * coeff)
            for trait, coeff in current_alphas.items()
            if abs(coeff) > 1e-6
        }
        return SafetyEvent(
            event_id=str(uuid4()),
            run_id=run_id,
            tick=tick,
            agent_id=agent_id,
            kind="toxicity",
            severity=severity,
            applied_alpha_delta=alpha_delta,
        )

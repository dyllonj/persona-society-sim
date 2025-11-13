from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, DefaultDict
from collections import defaultdict

from schemas.logs import ActionLog


GOALFUL_ACTIONS = {"research", "cite", "submit_report", "fill_field", "propose_plan", "submit_plan", "scan", "ping"}


@dataclass
class AgentMetrics:
    total_actions: int = 0
    goalful_actions: int = 0
    research_actions: int = 0
    cites: int = 0
    submit_tick: int | None = None
    first_action_tick: int | None = None
    collab_actions: int = 0

    def to_dict(self) -> Dict[str, object]:
        eff = (self.goalful_actions / self.total_actions) if self.total_actions else 0.0
        collab = (self.collab_actions / self.goalful_actions) if self.goalful_actions else 0.0
        duration = None
        if self.first_action_tick is not None and self.submit_tick is not None:
            duration = max(0, self.submit_tick - self.first_action_tick)
        return {
            "total_actions": self.total_actions,
            "goalful_actions": self.goalful_actions,
            "efficiency": round(eff, 3),
            "collab_ratio": round(collab, 3),
            "research_actions": self.research_actions,
            "cites": self.cites,
            "submit_tick": self.submit_tick,
            "time_to_submit": duration,
        }


class MetricTracker:
    def __init__(self, run_id: str, out_dir: Path = Path("metrics")) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.agent: DefaultDict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self.tick_collab_ratio: Dict[int, float] = {}
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_action(self, log: ActionLog, occupants: int | None = None) -> None:
        m = self.agent[log.agent_id]
        m.total_actions += 1
        if log.action_type in GOALFUL_ACTIONS:
            m.goalful_actions += 1
        if occupants is not None and log.action_type in {"talk", "trade", "work", "research", "scan"}:
            if occupants > 1:
                m.collab_actions += 1
        if m.first_action_tick is None:
            m.first_action_tick = log.tick
        if log.action_type == "research":
            m.research_actions += 1
        elif log.action_type == "cite":
            m.cites += 1
        elif log.action_type == "submit_report" and m.submit_tick is None:
            m.submit_tick = log.tick

    def on_tick_end(self, tick: int, collab_ratio: float) -> None:
        self.tick_collab_ratio[tick] = collab_ratio

    def flush(self) -> None:
        path = self.out_dir / f"run_{self.run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            summary = {
                "run_id": self.run_id,
                "tick_collab_ratio": self.tick_collab_ratio,
            }
            f.write(json.dumps({"summary": summary}) + "\n")
            for agent_id, m in sorted(self.agent.items()):
                f.write(json.dumps({"agent_id": agent_id, **m.to_dict()}) + "\n")

__all__ = ["MetricTracker"]


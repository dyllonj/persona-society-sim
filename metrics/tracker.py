from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Mapping
from collections import defaultdict

try:  # Optional dependency used when available for columnar exports
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except Exception:  # pragma: no cover - defensive optional dependency import
    pa = None
    pq = None
    _PARQUET_AVAILABLE = False

from metrics.persona_bands import BAND_METADATA, BAND_THRESHOLDS, TRAIT_ORDER
from schemas.agent import PersonaCoeffs
from schemas.logs import ActionLog, MsgLog


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
    trait_bands: Dict[str, str] = field(default_factory=dict)

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
            "trait_bands": dict(self.trait_bands),
        }


class MetricTracker:
    ALPHA_BUCKETS = (
        ("<0.5", 0.0, 0.5),
        ("0.5-1.5", 0.5, 1.5),
        (">1.5", 1.5, None),
    )

    def __init__(
        self,
        run_id: str,
        agent_personas: Mapping[str, PersonaCoeffs | Mapping[str, float]] | None = None,
        out_dir: Path = Path("metrics"),
    ) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.agent: DefaultDict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self.tick_collab_ratio: Dict[int, float] = {}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.agent_trait_bands: Dict[str, Dict[str, str]] = {}
        self.alpha_bucket_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.alpha_magnitude_totals: DefaultDict[str, float] = defaultdict(float)
        self.alpha_magnitude_counts: DefaultDict[str, int] = defaultdict(int)
        if agent_personas:
            self.register_personas(agent_personas)

    def register_personas(
        self, agent_personas: Mapping[str, PersonaCoeffs | Mapping[str, float]]
    ) -> None:
        for agent_id, persona in agent_personas.items():
            persona_dict = (
                persona.model_dump() if hasattr(persona, "model_dump") else dict(persona)
            )
            bands: Dict[str, str] = {}
            for trait in TRAIT_ORDER:
                value = persona_dict.get(trait)
                if value is None:
                    continue
                bands[trait] = self._band_for_value(float(value))
            self.agent_trait_bands[agent_id] = bands
            self.agent[agent_id].trait_bands = dict(bands)

    def _band_for_value(self, value: float) -> str:
        if value <= BAND_THRESHOLDS["low"]:
            return "low"
        if value >= BAND_THRESHOLDS["high"]:
            return "high"
        return "neutral"

    def _ensure_agent_registration(self, agent_id: str) -> AgentMetrics:
        metrics = self.agent[agent_id]
        if not metrics.trait_bands and agent_id in self.agent_trait_bands:
            metrics.trait_bands = dict(self.agent_trait_bands[agent_id])
        return metrics

    def on_action(self, log: ActionLog, occupants: int | None = None) -> None:
        m = self._ensure_agent_registration(log.agent_id)
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

    def on_message(self, msg: MsgLog) -> None:
        self._ensure_agent_registration(msg.from_agent)
        snapshot = msg.steering_snapshot or {}
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except json.JSONDecodeError:
                return
        for trait, value in snapshot.items():
            try:
                magnitude = abs(float(value))
            except (TypeError, ValueError):
                continue
            bucket = self._bucket_for_magnitude(magnitude)
            if bucket is None:
                continue
            self.alpha_bucket_counts[trait][bucket] += 1
            self.alpha_magnitude_totals[trait] += magnitude
            self.alpha_magnitude_counts[trait] += 1

    def flush(self) -> None:
        for agent_id in self.agent_trait_bands:
            self._ensure_agent_registration(agent_id)
        trait_aggregates = self._trait_band_aggregates()
        alpha_summary = self._alpha_bucket_summary()
        path = self.out_dir / f"run_{self.run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            summary = {
                "run_id": self.run_id,
                "tick_collab_ratio": self.tick_collab_ratio,
                "trait_band_aggregates": trait_aggregates,
                "trait_band_metadata": BAND_METADATA,
                "alpha_buckets": alpha_summary,
                "alpha_bucket_labels": self._alpha_bucket_metadata(),
            }
            f.write(json.dumps({"summary": summary}) + "\n")
            agent_rows = []
            for agent_id, m in sorted(self.agent.items()):
                record = {"agent_id": agent_id, **m.to_dict()}
                f.write(json.dumps(record) + "\n")
                agent_rows.append(record)
        self._write_parquet(f"run_{self.run_id}_agents.parquet", agent_rows)
        trait_rows = [
            {"trait_band": key, **values, "trait": key.split(":")[0], "band": key.split(":")[1]}
            for key, values in trait_aggregates.items()
        ]
        alpha_rows = [
            {
                "trait": trait,
                "avg_magnitude": stats.get("avg_magnitude", 0.0),
                "samples": stats.get("samples", 0),
                **{f"bucket::{label}": stats["bucket_counts"].get(label, 0)
                    for label, _, _ in self.ALPHA_BUCKETS},
            }
            for trait, stats in alpha_summary.items()
        ]
        self._write_parquet(f"run_{self.run_id}_trait_aggregates.parquet", trait_rows)
        self._write_parquet(f"run_{self.run_id}_alpha_aggregates.parquet", alpha_rows)

    # ---- aggregation helpers ----

    def _alpha_bucket_metadata(self) -> Dict[str, Dict[str, float]]:
        metadata: Dict[str, Dict[str, float]] = {}
        for label, lower, upper in self.ALPHA_BUCKETS:
            entry: Dict[str, float] = {}
            if lower is not None:
                entry["min"] = lower
            if upper is not None:
                entry["max"] = upper
            metadata[label] = entry
        return metadata

    def _bucket_for_magnitude(self, magnitude: float) -> str | None:
        for label, lower, upper in self.ALPHA_BUCKETS:
            lower_ok = lower is None or magnitude >= lower
            upper_ok = upper is None or magnitude < upper
            if lower_ok and upper_ok:
                return label
        return None

    def _alpha_bucket_summary(self) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for trait, bucket_counts in self.alpha_bucket_counts.items():
            counts = {label: bucket_counts.get(label, 0) for label, _, _ in self.ALPHA_BUCKETS}
            total = self.alpha_magnitude_totals.get(trait, 0.0)
            samples = self.alpha_magnitude_counts.get(trait, 0)
            avg = round(total / samples, 3) if samples else 0.0
            summary[trait] = {
                "bucket_counts": counts,
                "avg_magnitude": avg,
                "samples": samples,
            }
        return summary

    def _trait_band_aggregates(self) -> Dict[str, Dict[str, object]]:
        aggregates: Dict[str, Dict[str, object]] = {}
        for agent_id, metrics in self.agent.items():
            trait_bands = metrics.trait_bands or {}
            contributions = {
                "total_actions": metrics.total_actions,
                "goalful_actions": metrics.goalful_actions,
                "research_actions": metrics.research_actions,
                "cites": metrics.cites,
                "collab_actions": metrics.collab_actions,
                "submissions": 1 if metrics.submit_tick is not None else 0,
            }
            if metrics.first_action_tick is not None and metrics.submit_tick is not None:
                duration = max(0, metrics.submit_tick - metrics.first_action_tick)
            else:
                duration = 0
            contributions["time_to_submit_total"] = duration
            for trait, band in trait_bands.items():
                key = f"{trait}:{band}"
                if key not in aggregates:
                    aggregates[key] = {
                        "agent_ids": set(),
                        "total_actions": 0,
                        "goalful_actions": 0,
                        "research_actions": 0,
                        "cites": 0,
                        "collab_actions": 0,
                        "submissions": 0,
                        "time_to_submit_total": 0,
                    }
                bucket = aggregates[key]
                bucket["agent_ids"].add(agent_id)
                for field, value in contributions.items():
                    bucket[field] += value
        output: Dict[str, Dict[str, object]] = {}
        for key, bucket in aggregates.items():
            agent_count = len(bucket.pop("agent_ids"))
            total_actions = bucket.get("total_actions", 0)
            goalful = bucket.get("goalful_actions", 0)
            collab = bucket.get("collab_actions", 0)
            submissions = bucket.get("submissions", 0)
            time_total = bucket.pop("time_to_submit_total", 0)
            avg_time = (time_total / submissions) if submissions else None
            result = dict(bucket)
            result["agent_count"] = agent_count
            result["efficiency"] = round(goalful / total_actions, 3) if total_actions else 0.0
            result["collab_ratio"] = round(collab / goalful, 3) if goalful else 0.0
            result["submit_rate"] = round(submissions / agent_count, 3) if agent_count else 0.0
            result["avg_time_to_submit"] = avg_time
            output[key] = result
        return output

    def _write_parquet(self, filename: str, rows: list[Dict[str, object]]) -> None:
        if not rows or not _PARQUET_AVAILABLE:
            return
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, self.out_dir / filename)
        except Exception:
            pass

__all__ = ["MetricTracker"]


from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Mapping

try:  # Optional dependency used when available for columnar exports
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except Exception:  # pragma: no cover - defensive optional dependency import
    pa = None
    pq = None
    _PARQUET_AVAILABLE = False

from metrics.persona_bands import BAND_METADATA, BAND_THRESHOLDS, TRAIT_ORDER
from metrics.research import ResearchMetricAggregator
from schemas.agent import PersonaCoeffs
from schemas.logs import (
    ActionLog,
    MsgLog,
    CitationLog,
    PersonaStabilityLog,
    ReportGradeLog,
    ResearchFactLog,
)


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
    action_counts: Dict[str, int] = field(default_factory=dict)

    def action_distribution(self) -> Dict[str, float]:
        if not self.total_actions:
            return {}
        return {
            action_type: count / self.total_actions
            for action_type, count in sorted(self.action_counts.items())
        }

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
            "action_counts": dict(sorted(self.action_counts.items())),
            "action_distribution": self.action_distribution(),
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
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.agent_trait_bands: Dict[str, Dict[str, str]] = {}
        self.alpha_bucket_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.alpha_magnitude_totals: DefaultDict[str, float] = defaultdict(float)
        self.alpha_magnitude_counts: DefaultDict[str, int] = defaultdict(int)
        self.research_metrics = ResearchMetricAggregator()
        self.persona_stability_records: list[PersonaStabilityLog] = []
        self.mind_wander_injections: int = 0
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
        m.action_counts[log.action_type] = m.action_counts.get(log.action_type, 0) + 1
        if log.action_type in GOALFUL_ACTIONS:
            m.goalful_actions += 1
        if occupants is not None and log.action_type in {"talk", "work", "research", "scan"}:
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

    def on_research_fact(self, log: ResearchFactLog) -> None:
        self.research_metrics.observe_fact(log)

    def on_citation(self, log: CitationLog) -> None:
        self.research_metrics.observe_citation(log)

    def on_report_grade(self, log: ReportGradeLog) -> None:
        self.research_metrics.observe_grade(log)

    def on_persona_stability(self, log: PersonaStabilityLog | Mapping[str, object]) -> None:
        record = log if isinstance(log, PersonaStabilityLog) else PersonaStabilityLog(**dict(log))
        self._ensure_agent_registration(record.agent_id)
        self.persona_stability_records.append(record)

    def on_mind_wander(self, count: int = 1) -> None:
        self.mind_wander_injections += count

    def flush(self) -> None:
        for agent_id in self.agent_trait_bands:
            self._ensure_agent_registration(agent_id)
        trait_aggregates = self._trait_band_aggregates()
        alpha_summary = self._alpha_bucket_summary()
        research_summary = self.research_metrics.summary()
        passive_validity = self._passive_validity_summary()
        path = self.out_dir / f"run_{self.run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            summary = {
                "run_id": self.run_id,
                "tick_collab_ratio": self.tick_collab_ratio,
                "trait_band_aggregates": trait_aggregates,
                "trait_band_metadata": BAND_METADATA,
                "alpha_buckets": alpha_summary,
                "alpha_bucket_labels": self._alpha_bucket_metadata(),
                "research": research_summary,
                "passive_validity": passive_validity,
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

    def population_trait_variance(self) -> Dict[str, float]:
        """Population variance of per-agent mean probe scores for each trait."""

        return {
            trait: self._round_metric(self._population_variance(agent_means.values()))
            for trait, agent_means in sorted(self._trait_agent_means().items())
        }

    def behavioral_variance(self) -> Dict[str, object]:
        """Variance of action-type proportions across agent action distributions."""

        distributions = self._agent_action_distributions()
        action_types = sorted(
            {
                action_type
                for distribution in distributions.values()
                for action_type in distribution
            }
        )
        by_action: Dict[str, float] = {}
        ratios: Dict[str, float] = {}
        for action_type in action_types:
            values = [
                distribution.get(action_type, 0.0)
                for distribution in distributions.values()
            ]
            by_action[action_type] = self._round_metric(self._population_variance(values))
            ratios[action_type] = self.variance_vs_mean_ratio(values)
        overall = (
            self._round_metric(sum(by_action.values()) / len(by_action))
            if by_action
            else 0.0
        )
        return {
            "agent_count": len(distributions),
            "action_distributions": distributions,
            "by_action": by_action,
            "overall": overall,
            "variance_vs_mean_ratio": ratios,
        }

    @staticmethod
    def variance_vs_mean_ratio(values: Iterable[float]) -> float:
        numbers = MetricTracker._finite_numbers(values)
        if not numbers:
            return 0.0
        mean = sum(numbers) / len(numbers)
        if math.isclose(mean, 0.0, abs_tol=1e-12):
            return 0.0
        return MetricTracker._round_metric(
            MetricTracker._population_variance(numbers) / abs(mean)
        )

    def cronbachs_alpha(self) -> Dict[str, float | None]:
        """Cronbach's alpha by trait using probe text as the item dimension."""

        output: Dict[str, float | None] = {}
        for trait, (items, rows) in sorted(self._trait_item_matrices().items()):
            alpha = self._cronbach_alpha_matrix(items, rows)
            output[trait] = self._round_metric(alpha) if alpha is not None else None
        return output

    def test_retest_stability(self) -> Dict[str, float | None]:
        """Correlation of first vs latest repeated probe scores by trait."""

        grouped: DefaultDict[str, DefaultDict[tuple[str, str], list[tuple[int, float]]]]
        grouped = defaultdict(lambda: defaultdict(list))
        for record in self.persona_stability_records:
            for trait, score in self._finite_trait_scores(record).items():
                grouped[trait][(record.agent_id, record.probe_text)].append((record.tick, score))

        output: Dict[str, float | None] = {}
        for trait, probe_groups in sorted(grouped.items()):
            first_scores: list[float] = []
            latest_scores: list[float] = []
            for observations in probe_groups.values():
                if len(observations) < 2:
                    continue
                observations.sort(key=lambda item: item[0])
                first_scores.append(observations[0][1])
                latest_scores.append(observations[-1][1])
            if not first_scores:
                output[trait] = None
                continue
            if len(first_scores) == 1:
                output[trait] = 1.0 if math.isclose(first_scores[0], latest_scores[0]) else 0.0
                continue
            correlation = self._pearson(first_scores, latest_scores)
            if correlation is None and all(
                math.isclose(first, latest)
                for first, latest in zip(first_scores, latest_scores)
            ):
                correlation = 1.0
            output[trait] = self._round_metric(correlation) if correlation is not None else None
        return output

    # ---- passive validity helpers ----

    def _passive_validity_summary(self) -> Dict[str, object]:
        behavioral = self.behavioral_variance()
        return {
            "samples": len(self.persona_stability_records),
            "population_trait_variance": self.population_trait_variance(),
            "behavioral_variance": behavioral,
            "variance_vs_mean_ratio": {
                "traits": self._trait_variance_vs_mean_ratio(),
                "actions": behavioral.get("variance_vs_mean_ratio", {}),
            },
            "cronbachs_alpha": self.cronbachs_alpha(),
            "test_retest_stability": self.test_retest_stability(),
            "embedding_distance_from_baseline": self._embedding_distance_summary(),
            "mind_wander_injections": self.mind_wander_injections,
        }

    def _agent_action_distributions(self) -> Dict[str, Dict[str, float]]:
        return {
            agent_id: metrics.action_distribution()
            for agent_id, metrics in sorted(self.agent.items())
        }

    def _trait_agent_means(self) -> Dict[str, Dict[str, float]]:
        values: DefaultDict[str, DefaultDict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self.persona_stability_records:
            for trait, score in self._finite_trait_scores(record).items():
                values[trait][record.agent_id].append(score)
        return {
            trait: {
                agent_id: sum(scores) / len(scores)
                for agent_id, scores in sorted(agent_values.items())
                if scores
            }
            for trait, agent_values in sorted(values.items())
        }

    def _trait_variance_vs_mean_ratio(self) -> Dict[str, float]:
        return {
            trait: self.variance_vs_mean_ratio(agent_means.values())
            for trait, agent_means in sorted(self._trait_agent_means().items())
        }

    def _trait_item_matrices(
        self,
    ) -> Dict[str, tuple[list[str], list[list[float]]]]:
        values: DefaultDict[
            str,
            DefaultDict[str, DefaultDict[str, list[float]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for record in self.persona_stability_records:
            for trait, score in self._finite_trait_scores(record).items():
                values[trait][record.agent_id][record.probe_text].append(score)

        matrices: Dict[str, tuple[list[str], list[list[float]]]] = {}
        for trait, by_agent in sorted(values.items()):
            items = sorted(
                {probe_text for item_values in by_agent.values() for probe_text in item_values}
            )
            rows: list[list[float]] = []
            if items:
                for item_values in by_agent.values():
                    if all(item in item_values for item in items):
                        rows.append(
                            [
                                sum(item_values[item]) / len(item_values[item])
                                for item in items
                            ]
                        )
            matrices[trait] = (items, rows)
        return matrices

    @staticmethod
    def _cronbach_alpha_matrix(items: list[str], rows: list[list[float]]) -> float | None:
        item_count = len(items)
        if item_count < 2 or len(rows) < 2:
            return None
        item_variances = [
            MetricTracker._sample_variance(row[column] for row in rows)
            for column in range(item_count)
        ]
        total_scores = [sum(row) for row in rows]
        total_variance = MetricTracker._sample_variance(total_scores)
        if math.isclose(total_variance, 0.0, abs_tol=1e-12):
            return None
        return (item_count / (item_count - 1)) * (
            1.0 - (sum(item_variances) / total_variance)
        )

    def _embedding_distance_summary(self) -> Dict[str, float | int | None]:
        distances = self._finite_numbers(
            record.embedding_distance_from_baseline
            for record in self.persona_stability_records
        )
        if not distances:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(distances),
            "mean": self._round_metric(sum(distances) / len(distances)),
            "min": self._round_metric(min(distances)),
            "max": self._round_metric(max(distances)),
        }

    @staticmethod
    def _finite_trait_scores(record: PersonaStabilityLog) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for trait, value in (record.trait_scores or {}).items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores[trait] = score
        return scores

    @staticmethod
    def _finite_numbers(values: Iterable[float]) -> list[float]:
        numbers: list[float] = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(number)
        return numbers

    @staticmethod
    def _population_variance(values: Iterable[float]) -> float:
        numbers = MetricTracker._finite_numbers(values)
        if not numbers:
            return 0.0
        mean = sum(numbers) / len(numbers)
        return sum((value - mean) ** 2 for value in numbers) / len(numbers)

    @staticmethod
    def _sample_variance(values: Iterable[float]) -> float:
        numbers = MetricTracker._finite_numbers(values)
        if len(numbers) < 2:
            return 0.0
        mean = sum(numbers) / len(numbers)
        return sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denom_x = sum((x - mean_x) ** 2 for x in xs)
        denom_y = sum((y - mean_y) ** 2 for y in ys)
        denominator = math.sqrt(denom_x * denom_y)
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            return None
        return numerator / denominator

    @staticmethod
    def _round_metric(value: float) -> float:
        rounded = round(float(value), 6)
        return 0.0 if math.isclose(rounded, 0.0, abs_tol=1e-12) else rounded

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
                for metric_name, value in contributions.items():
                    bucket[metric_name] += value
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

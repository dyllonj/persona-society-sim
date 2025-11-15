"""Structured logging to DB and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency for structured dumps
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pa = None
    pq = None

from schemas.logs import (
    ActionLog,
    GraphSnapshot,
    MetricsSnapshot,
    MsgLog,
    SafetyEvent,
    CitationLog,
    ResearchFactLog,
    ReportGradeLog,
    ProbeLog,
    BehaviorProbeLog,
)
from storage.db import Database


class LogSink:
    def __init__(self, run_id: str, db_url: str | None, parquet_dir: str | None):
        self.run_id = run_id
        self.db = Database(db_url) if db_url else None
        if self.db:
            self.db.init()
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        if self.parquet_dir:
            for sub in (
                "actions",
                "messages",
                "safety",
                "graph_snapshots",
                "metrics_snapshots",
                "research_facts",
                "citations",
                "report_grades",
                "probe_logs",
                "behavior_probes",
            ):
                (self.parquet_dir / sub).mkdir(parents=True, exist_ok=True)
        self.action_buffer: List[ActionLog] = []
        self.msg_buffer: List[MsgLog] = []
        self.safety_buffer: List[SafetyEvent] = []
        self.graph_buffer: List[GraphSnapshot] = []
        self.metrics_buffer: List[MetricsSnapshot] = []
        self.research_buffer: List[ResearchFactLog] = []
        self.citation_buffer: List[CitationLog] = []
        self.report_grade_buffer: List[ReportGradeLog] = []
        self.probe_buffer: List[ProbeLog] = []
        self.behavior_probe_buffer: List[BehaviorProbeLog] = []

    def log_action(self, log: ActionLog) -> None:
        self.action_buffer.append(log)

    def log_message(self, log: MsgLog) -> None:
        self.msg_buffer.append(log)

    def log_safety(self, event: SafetyEvent) -> None:
        self.safety_buffer.append(event)

    def log_graph_snapshot(self, snapshot: GraphSnapshot) -> None:
        self.graph_buffer.append(snapshot)

    def log_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        self.metrics_buffer.append(snapshot)

    def log_research_fact(self, log: ResearchFactLog) -> None:
        self.research_buffer.append(log)

    def log_citation(self, log: CitationLog) -> None:
        self.citation_buffer.append(log)

    def log_report_grade(self, log: ReportGradeLog) -> None:
        self.report_grade_buffer.append(log)

    def log_probe(self, log: ProbeLog) -> None:
        self.probe_buffer.append(log)

    def log_behavior_probe(self, log: BehaviorProbeLog) -> None:
        self.behavior_probe_buffer.append(log)

    def flush(self, tick: int) -> None:
        self._flush_buffer("action_log", self.action_buffer)
        self._flush_buffer("msg_log", self.msg_buffer)
        self._flush_buffer("safety_event", self.safety_buffer)
        self._flush_buffer("graph_snapshot", self.graph_buffer)
        self._flush_buffer("metrics_snapshot", self.metrics_buffer)
        self._flush_buffer("research_fact_log", self.research_buffer)
        self._flush_buffer("citation_log", self.citation_buffer)
        self._flush_buffer("report_grade_log", self.report_grade_buffer)
        self._flush_buffer("probe_log", self.probe_buffer)
        self._flush_buffer("behavior_probe_log", self.behavior_probe_buffer)
        if self.parquet_dir:
            self._write_parquet(self.action_buffer, "actions", tick)
            self._write_parquet(self.msg_buffer, "messages", tick)
            self._write_parquet(self.safety_buffer, "safety", tick)
            self._write_parquet(self.graph_buffer, "graph_snapshots", tick)
            self._write_parquet(self.metrics_buffer, "metrics_snapshots", tick)
            self._write_parquet(self.research_buffer, "research_facts", tick)
            self._write_parquet(self.citation_buffer, "citations", tick)
            self._write_parquet(self.report_grade_buffer, "report_grades", tick)
            self._write_parquet(self.probe_buffer, "probe_logs", tick)
            self._write_parquet(self.behavior_probe_buffer, "behavior_probes", tick)
        self.action_buffer.clear()
        self.msg_buffer.clear()
        self.safety_buffer.clear()
        self.graph_buffer.clear()
        self.metrics_buffer.clear()
        self.research_buffer.clear()
        self.citation_buffer.clear()
        self.report_grade_buffer.clear()
        self.probe_buffer.clear()
        self.behavior_probe_buffer.clear()

    # ---- internals ----

    def _flush_buffer(self, table: str, buffer: Sequence) -> None:
        if not buffer or not self.db:
            return
        rows = [self._normalize(record.model_dump()) for record in buffer]
        self.db.insert_many(table, rows)

    def _write_parquet(self, buffer: Sequence, kind: str, tick: int) -> None:
        if not buffer or pa is None or pq is None:
            return
        rows = [self._normalize(record.model_dump()) for record in buffer]
        table = pa.Table.from_pylist(rows)
        path = self.parquet_dir / kind / f"{kind}_t{tick:05d}.parquet"
        pq.write_table(table, path)

    @staticmethod
    def _normalize(record: Dict) -> Dict:
        normalized = {}
        for key, value in record.items():
            if key == "trait_key" and value is None:
                normalized[key] = "global"
                continue
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = value
        return normalized

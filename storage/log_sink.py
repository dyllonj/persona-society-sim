"""Structured logging to DB and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from schemas.logs import ActionLog, MsgLog, SafetyEvent
from storage.db import Database


class LogSink:
    def __init__(self, run_id: str, db_url: str | None, parquet_dir: str | None):
        self.run_id = run_id
        self.db = Database(db_url) if db_url else None
        if self.db:
            self.db.init()
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        if self.parquet_dir:
            for sub in ("actions", "messages", "safety"):
                (self.parquet_dir / sub).mkdir(parents=True, exist_ok=True)
        self.action_buffer: List[ActionLog] = []
        self.msg_buffer: List[MsgLog] = []
        self.safety_buffer: List[SafetyEvent] = []

    def log_action(self, log: ActionLog) -> None:
        self.action_buffer.append(log)

    def log_message(self, log: MsgLog) -> None:
        self.msg_buffer.append(log)

    def log_safety(self, event: SafetyEvent) -> None:
        self.safety_buffer.append(event)

    def flush(self, tick: int) -> None:
        self._flush_buffer("action_log", self.action_buffer)
        self._flush_buffer("msg_log", self.msg_buffer)
        self._flush_buffer("safety_event", self.safety_buffer)
        if self.parquet_dir:
            self._write_parquet(self.action_buffer, "actions", tick)
            self._write_parquet(self.msg_buffer, "messages", tick)
            self._write_parquet(self.safety_buffer, "safety", tick)
        self.action_buffer.clear()
        self.msg_buffer.clear()
        self.safety_buffer.clear()

    # ---- internals ----

    def _flush_buffer(self, table: str, buffer: Sequence) -> None:
        if not buffer or not self.db:
            return
        rows = [self._normalize(record.model_dump()) for record in buffer]
        self.db.insert_many(table, rows)

    def _write_parquet(self, buffer: Sequence, kind: str, tick: int) -> None:
        if not buffer:
            return
        rows = [self._normalize(record.model_dump()) for record in buffer]
        table = pa.Table.from_pylist(rows)
        path = self.parquet_dir / kind / f"{kind}_t{tick:05d}.parquet"
        pq.write_table(table, path)

    @staticmethod
    def _normalize(record: Dict) -> Dict:
        normalized = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = value
        return normalized

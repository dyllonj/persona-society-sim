"""Queue-backed runtime adapters for scaling the simulation loop.

These adapters provide a migration seam before the runner is fully refactored
into a queue-native pipeline. They preserve the current ``SimulationRunner``
interfaces: ``LogSink``-style log methods plus ``flush(tick)``, and
event-bridge ``start``/``broadcast``/``stop`` methods.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from schemas.logs import (
    ActionLog,
    BehaviorProbeLog,
    CitationLog,
    GraphSnapshot,
    MetricsSnapshot,
    MsgLog,
    ProbeLog,
    ReportGradeLog,
    ResearchFactLog,
    SafetyEvent,
)
from storage.log_sink import LogSink


@dataclass
class QueueRuntimeStats:
    enqueued_records: int = 0
    flushes: int = 0
    broadcasts: int = 0
    dropped_broadcasts: int = 0
    worker_errors: int = 0


class QueueBackedLogSink(LogSink):
    """LogSink variant that serializes writes through a worker queue."""

    def __init__(
        self,
        run_id: str,
        db_url: str | None,
        parquet_dir: str | None,
        *,
        maxsize: int = 10000,
        wait_on_flush: bool = True,
    ) -> None:
        super().__init__(run_id, db_url, parquet_dir)
        self.wait_on_flush = wait_on_flush
        self.stats = QueueRuntimeStats()
        self._queue: queue.Queue[tuple[str, Any, Optional[threading.Event]]] = queue.Queue(
            maxsize=max(1, maxsize)
        )
        self._closed = False
        self._errors: list[str] = []
        self._worker = threading.Thread(
            target=self._run_worker,
            name=f"log-sink-{run_id}",
            daemon=True,
        )
        self._worker.start()

    @property
    def worker_errors(self) -> list[str]:
        return list(self._errors)

    def log_action(self, log: ActionLog) -> None:
        self._enqueue("action", log)

    def log_message(self, log: MsgLog) -> None:
        self._enqueue("message", log)

    def log_safety(self, event: SafetyEvent) -> None:
        self._enqueue("safety", event)

    def log_graph_snapshot(self, snapshot: GraphSnapshot) -> None:
        self._enqueue("graph", snapshot)

    def log_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        self._enqueue("metrics", snapshot)

    def log_research_fact(self, log: ResearchFactLog) -> None:
        self._enqueue("research", log)

    def log_citation(self, log: CitationLog) -> None:
        self._enqueue("citation", log)

    def log_report_grade(self, log: ReportGradeLog) -> None:
        self._enqueue("report_grade", log)

    def log_probe(self, log: ProbeLog) -> None:
        self._enqueue("probe", log)

    def log_behavior_probe(self, log: BehaviorProbeLog) -> None:
        self._enqueue("behavior_probe", log)

    def flush(self, tick: int) -> None:
        done = threading.Event()
        self._queue.put(("flush", tick, done))
        if self.wait_on_flush:
            done.wait()
            self._raise_worker_error_if_needed()

    def close(self) -> None:
        if self._closed:
            return
        done = threading.Event()
        self._queue.put(("close", None, done))
        done.wait()
        self._worker.join(timeout=5)
        self._closed = True
        self._raise_worker_error_if_needed()

    def _enqueue(self, kind: str, record: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot log to a closed QueueBackedLogSink")
        self._queue.put((kind, record, None))
        self.stats.enqueued_records += 1

    def _run_worker(self) -> None:
        while True:
            kind, payload, done = self._queue.get()
            try:
                if kind == "close":
                    self._flush_remaining()
                    if done:
                        done.set()
                    return
                if kind == "flush":
                    LogSink.flush(self, int(payload))
                    self.stats.flushes += 1
                    if done:
                        done.set()
                    continue
                self._dispatch_record(kind, payload)
            except Exception:
                self.stats.worker_errors += 1
                self._errors.append(traceback.format_exc())
                if done:
                    done.set()
            finally:
                self._queue.task_done()

    def _dispatch_record(self, kind: str, payload: Any) -> None:
        if kind == "action":
            LogSink.log_action(self, payload)
        elif kind == "message":
            LogSink.log_message(self, payload)
        elif kind == "safety":
            LogSink.log_safety(self, payload)
        elif kind == "graph":
            LogSink.log_graph_snapshot(self, payload)
        elif kind == "metrics":
            LogSink.log_metrics_snapshot(self, payload)
        elif kind == "research":
            LogSink.log_research_fact(self, payload)
        elif kind == "citation":
            LogSink.log_citation(self, payload)
        elif kind == "report_grade":
            LogSink.log_report_grade(self, payload)
        elif kind == "probe":
            LogSink.log_probe(self, payload)
        elif kind == "behavior_probe":
            LogSink.log_behavior_probe(self, payload)
        else:
            raise ValueError(f"unknown queued log kind: {kind}")

    def _flush_remaining(self) -> None:
        has_buffers = any(
            (
                self.action_buffer,
                self.msg_buffer,
                self.safety_buffer,
                self.graph_buffer,
                self.metrics_buffer,
                self.research_buffer,
                self.citation_buffer,
                self.report_grade_buffer,
                self.probe_buffer,
                self.behavior_probe_buffer,
            )
        )
        if has_buffers:
            LogSink.flush(self, -1)

    def _raise_worker_error_if_needed(self) -> None:
        if self._errors:
            raise RuntimeError(self._errors[-1])


class QueuedEventBridge:
    """Async broadcast wrapper for viewer/TUI bridges."""

    def __init__(
        self,
        bridge: Any,
        *,
        maxsize: int = 10000,
        drop_on_full: bool = True,
        start_underlying: bool = True,
        stop_underlying: bool = True,
    ) -> None:
        self.bridge = bridge
        self.drop_on_full = drop_on_full
        self.start_underlying = start_underlying
        self.stop_underlying = stop_underlying
        self.stats = QueueRuntimeStats()
        self._queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=max(1, maxsize))
        self._started = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="event-bridge-broadcast",
            daemon=True,
        )

    def start(self) -> None:
        if self._started:
            return
        if self.start_underlying and hasattr(self.bridge, "start"):
            self.bridge.start()
        self._started = True
        self._worker.start()

    def broadcast(self, payload: Any) -> None:
        if not self._started:
            self.start()
        try:
            if self.drop_on_full:
                self._queue.put_nowait(payload)
            else:
                self._queue.put(payload)
            self.stats.broadcasts += 1
        except queue.Full:
            self.stats.dropped_broadcasts += 1

    def stop(self) -> None:
        if not self._started:
            if self.stop_underlying and hasattr(self.bridge, "stop"):
                self.bridge.stop()
            return
        self._queue.put(None)
        self._worker.join(timeout=5)
        self._started = False
        if self.stop_underlying and hasattr(self.bridge, "stop"):
            self.bridge.stop()

    def _run_worker(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                self.bridge.broadcast(payload)
            except Exception:
                self.stats.worker_errors += 1
            finally:
                self._queue.task_done()


__all__ = ["QueueBackedLogSink", "QueuedEventBridge", "QueueRuntimeStats"]

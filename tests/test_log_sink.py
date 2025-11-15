from __future__ import annotations

from pathlib import Path

from schemas.logs import ActionLog, Edge, GraphSnapshot, MetricsSnapshot
from storage import log_sink as log_sink_module
from storage.log_sink import LogSink


def test_log_sink_flush(tmp_path: Path):
    sink = LogSink(run_id="r1", db_url=None, parquet_dir=str(tmp_path))
    action = ActionLog(
        action_id="a1",
        run_id="r1",
        tick=0,
        agent_id="agent-1",
        action_type="talk",
        params={"topic": "test"},
        outcome="success",
        info={"utterance": "hi"},
        prompt_text="prompt",
        prompt_hash="abc123",
        plan_metadata={"action_type": "talk"},
        reflection_summary="summary",
        reflection_implications=["imp"],
    )
    sink.log_action(action)
    graph = GraphSnapshot(
        run_id="r1",
        tick=0,
        edges=[Edge(src="agent-1", dst="agent-2", weight=1.0, kind="message")],
        centrality={"agent-1": 1.0},
    )
    metrics = MetricsSnapshot(
        run_id="r1",
        tick=0,
        cooperation_rate=1.0,
        gini_wealth=0.0,
        polarization_modularity=0.0,
        conflicts=0,
        rule_enforcement_cost=0.0,
    )
    sink.log_graph_snapshot(graph)
    sink.log_metrics_snapshot(metrics)
    assert sink.action_buffer and sink.graph_buffer and sink.metrics_buffer
    sink.flush(tick=0)
    assert not sink.action_buffer
    assert not sink.graph_buffer
    assert not sink.metrics_buffer
    graph_dir = tmp_path / "graph_snapshots"
    metrics_dir = tmp_path / "metrics_snapshots"
    assert graph_dir.exists() and metrics_dir.exists()
    if log_sink_module.pq is not None:
        graph_files = list(graph_dir.glob("*.parquet"))
        metrics_files = list(metrics_dir.glob("*.parquet"))
        assert graph_files and metrics_files

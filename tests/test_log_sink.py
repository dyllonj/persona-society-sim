from __future__ import annotations

from pathlib import Path

from schemas.logs import (
    ActionLog,
    CitationLog,
    Edge,
    GraphSnapshot,
    MetricsSnapshot,
    ReportGradeLog,
    ResearchFactLog,
)
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
    fact_log = ResearchFactLog(
        log_id="rf1",
        run_id="r1",
        tick=0,
        agent_id="agent-1",
        doc_id="doc-1",
        fact_id="fact-1",
        fact_answer="answer",
        target_answer="answer",
        correct=True,
    )
    citation = CitationLog(
        log_id="c1",
        run_id="r1",
        tick=0,
        agent_id="agent-1",
        doc_id="doc-1",
    )
    report = ReportGradeLog(
        log_id="g1",
        run_id="r1",
        tick=0,
        agent_id="agent-1",
        targets_total=3,
        facts_correct=2,
        citations_valid=1,
        reward_points=2.5,
    )
    sink.log_graph_snapshot(graph)
    sink.log_metrics_snapshot(metrics)
    sink.log_research_fact(fact_log)
    sink.log_citation(citation)
    sink.log_report_grade(report)
    assert sink.action_buffer and sink.graph_buffer and sink.metrics_buffer
    assert sink.research_buffer and sink.citation_buffer and sink.report_grade_buffer
    sink.flush(tick=0)
    assert not sink.action_buffer
    assert not sink.graph_buffer
    assert not sink.metrics_buffer
    assert not sink.research_buffer
    assert not sink.citation_buffer
    assert not sink.report_grade_buffer
    graph_dir = tmp_path / "graph_snapshots"
    metrics_dir = tmp_path / "metrics_snapshots"
    research_dir = tmp_path / "research_facts"
    citation_dir = tmp_path / "citations"
    grades_dir = tmp_path / "report_grades"
    assert graph_dir.exists() and metrics_dir.exists()
    assert research_dir.exists() and citation_dir.exists() and grades_dir.exists()
    if log_sink_module.pq is not None:
        graph_files = list(graph_dir.glob("*.parquet"))
        metrics_files = list(metrics_dir.glob("*.parquet"))
        fact_files = list(research_dir.glob("*.parquet"))
        citation_files = list(citation_dir.glob("*.parquet"))
        grade_files = list(grades_dir.glob("*.parquet"))
        assert graph_files and metrics_files
        assert fact_files and citation_files and grade_files

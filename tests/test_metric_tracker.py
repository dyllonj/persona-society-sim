import json
from pathlib import Path

from metrics.tracker import MetricTracker
from schemas.agent import PersonaCoeffs
from schemas.logs import (
    ActionLog,
    CitationLog,
    MsgLog,
    ReportGradeLog,
    ResearchFactLog,
)


def _action(agent_id: str, tick: int, action_type: str) -> ActionLog:
    return ActionLog(
        action_id=f"a-{tick}",
        run_id="unit",
        tick=tick,
        agent_id=agent_id,
        action_type=action_type,
        params={},
        outcome="success",
        info={},
    )


def _message(agent_id: str, tick: int, snapshot: dict[str, float]) -> MsgLog:
    return MsgLog(
        msg_id=f"m-{tick}",
        run_id="unit",
        tick=tick,
        channel="room",
        from_agent=agent_id,
        to_agent=None,
        room_id="commons",
        content="hello",
        tokens_in=5,
        tokens_out=10,
        temperature=0.1,
        top_p=0.9,
        steering_snapshot=snapshot,
        layers_used=[1],
    )


def test_metric_tracker_handles_personas_and_messages(tmp_path: Path) -> None:
    personas = {
        "agent-1": PersonaCoeffs(E=0.2, A=-2.0, C=1.6, O=0.0, N=0.1),
    }
    tracker = MetricTracker("unit", agent_personas=personas, out_dir=tmp_path)
    tracker.on_action(_action("agent-1", 1, "research"), occupants=2)
    tracker.on_tick_end(1, 0.5)
    tracker.on_message(_message("agent-1", 1, {"E": 0.8, "A": -2.1}))
    tracker.on_research_fact(
        ResearchFactLog(
            log_id="rf-1",
            run_id="unit",
            tick=1,
            agent_id="agent-1",
            doc_id="doc-1",
            fact_id="fact-1",
            fact_answer="alpha",
            target_answer="alpha",
            correct=True,
            trait_key="A:low",
        )
    )
    tracker.on_citation(
        CitationLog(
            log_id="c-1",
            run_id="unit",
            tick=1,
            agent_id="agent-1",
            doc_id="doc-1",
            trait_key="A:low",
        )
    )
    tracker.on_report_grade(
        ReportGradeLog(
            log_id="g-1",
            run_id="unit",
            tick=1,
            agent_id="agent-1",
            targets_total=3,
            facts_correct=2,
            citations_valid=1,
            reward_points=3.0,
            trait_key="A:low",
        )
    )
    tracker.flush()

    log_path = tmp_path / "run_unit.jsonl"
    assert log_path.exists(), "flush should emit a jsonl file"

    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    summary = lines[0]["summary"]
    agent_payload = next(line for line in lines if line.get("agent_id") == "agent-1")

    assert agent_payload["trait_bands"]["A"] == "low"
    assert agent_payload["total_actions"] == 1

    trait_aggs = summary["trait_band_aggregates"]
    assert trait_aggs["A:low"]["total_actions"] == 1

    alpha_buckets = summary["alpha_buckets"]
    assert alpha_buckets["E"]["bucket_counts"]["0.5-1.5"] == 1
    assert alpha_buckets["A"]["bucket_counts"][">1.5"] == 1
    research = summary["research"]
    assert research["fact_coverage"]["A:low"]["facts_correct"] == 1.0
    assert research["citation_diversity"]["A:low"]["total"] == 1.0
    assert research["grade_drift"]["A:low"]["avg_reward"] == 3.0

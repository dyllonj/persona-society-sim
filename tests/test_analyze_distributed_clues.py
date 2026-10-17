import json

import pytest

from scripts.analyze_distributed_clues import (
    classify_submissions,
    load_batches,
    plurality_vote,
    summarize,
    variance_components,
)


def _sub(correct, complete, valid=True, plan=None):
    return {"valid": valid, "answer_correct": correct, "evidence_complete": complete,
            "plan_id": plan or ("answer" if correct else "wrong")}


def _row(seed, condition, correct, replicate=0, coverage=None, submissions=None):
    coverage = coverage or {"a": 1.0, "b": 1.0, "c": 1.0}
    submissions = submissions or {r: _sub(i < correct, i < correct) for i, r in enumerate(coverage)}
    return {
        "seed": seed, "replicate": replicate, "condition": condition, "agents": len(coverage),
        "correct_answers": correct, "supported_submissions": correct, "team_success": correct > 0,
        "invalid_actions": 0, "turns_executed": 12, "evidence_coverage": coverage,
        "submissions": submissions, "tokens_in": 10, "tokens_out": 1, "wall_seconds": 1.0,
    }


def _batch(path, rows, status="complete", prompt="p1"):
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps(
        {"status": status, "task_version": "t1", "prompt_version": prompt, "model_revision": "m1"}))
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rejects_incomplete_batches(tmp_path):
    batch = _batch(tmp_path / "a", [_row(1, "communication", 3)], status="incomplete")
    with pytest.raises(ValueError, match="not a complete batch"):
        load_batches([batch])


def test_rejects_mixed_prompt_versions_and_duplicates(tmp_path):
    a = _batch(tmp_path / "a", [_row(1, "communication", 3)])
    b = _batch(tmp_path / "b", [_row(2, "communication", 3)], prompt="p2")
    with pytest.raises(ValueError, match="mixes"):
        load_batches([a, b])
    c = _batch(tmp_path / "c", [_row(1, "communication", 2)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_batches([a, c])


def test_classification_separates_delivery_from_synthesis_failures():
    row = _row(1, "communication", 1, coverage={"a": 1.0, "b": 1.0, "c": 1 / 3, "d": 1.0, "e": 1.0},
               submissions={"a": _sub(True, True), "b": _sub(False, True), "c": _sub(False, False),
                            "d": _sub(True, False), "e": _sub(False, False, valid=False)})
    assert dict(classify_submissions(row)) == {
        "correct_supported": 1, "wrong_despite_all_clues": 1, "wrong_missing_clues": 1,
        "correct_but_evidence_not_cited": 1, "invalid_or_missing": 1}


def test_paired_effect_uses_instances_not_replicates():
    rows = [_row(seed, "communication", 3, replicate=r) for seed in (1, 2) for r in (0, 1)]
    rows += [_row(1, "no_communication", 0, replicate=r) for r in (0, 1)]
    rows += [_row(2, "no_communication", 3, replicate=0), _row(2, "no_communication", 0, replicate=1)]
    paired = summarize(rows)["paired_communication_effect"]
    assert paired["instances"] == 2
    assert paired["mean_difference"] == pytest.approx(0.75)
    assert paired["instances_favoring_communication"] == 2


def test_variance_components_attribute_spread_to_instances():
    rows = [_row(1, "no_communication", 3, replicate=r) for r in (0, 1)]
    rows += [_row(2, "no_communication", 0, replicate=r) for r in (0, 1)]
    components = variance_components(rows)
    assert components["within_instance_variance"] == 0.0
    assert components["instance_share"] == pytest.approx(1.0)


def test_plurality_vote_strict_and_tie_break():
    def sub(plan, correct):
        return {"valid": True, "plan_id": plan, "answer_correct": correct, "evidence_complete": False}
    majority = _row(1, "no_communication", 2, submissions={"a": sub("p1", True), "b": sub("p1", True), "c": sub("p2", False)})
    tie = _row(1, "no_communication", 1, submissions={"a": sub("p1", True), "b": sub("p2", False), "c": sub("p3", False)})
    assert plurality_vote(majority) == (1.0, 1.0)
    assert plurality_vote(tie) == (0.0, pytest.approx(1 / 3))


def test_guess_diagnostics_reports_candidate_position_and_chance(tmp_path):
    from scripts.analyze_distributed_clues import guess_diagnostics

    episode = tmp_path / "batch" / "seed-1-rep-0-no_communication"
    episode.mkdir(parents=True)
    clues = {"e1": {"allowed_plans": ["plan_2", "plan_4", "plan_6", "plan_8"]},
             "e2": {"allowed_plans": ["plan_1", "plan_2", "plan_3", "plan_4"]},
             "e3": {"allowed_plans": ["plan_4", "plan_5", "plan_6", "plan_7"]}}
    (episode / "evaluator-only.json").write_text(json.dumps({"clues": clues, "answer": "plan_4"}))
    (episode / "summary.json").write_text(json.dumps({"condition": "no_communication", "submissions": {
        "a": {"valid": True, "plan_id": "plan_2"}, "b": {"valid": True, "plan_id": "plan_9"}}}))
    trace = [{"agent": "a", "phase": "submit", "known_before": ["e1"]},
             {"agent": "b", "phase": "submit", "known_before": ["e1", "e2"]}]
    (episode / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    result = guess_diagnostics([tmp_path / "batch"])
    assert result["partial_information_submissions"] == 2
    assert result["consistent_with_held_clues"] == 0.5
    assert result["candidate_position_counts"] == {0: 1}
    assert result["uniform_guess_expected_accuracy"] == pytest.approx((1 / 4 + 1 / 2) / 2)

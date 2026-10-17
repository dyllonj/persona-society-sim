"""Offline audit regressions; scripted policies are not model evidence."""
import copy
import hashlib
import json
from urllib.error import HTTPError, URLError

import pytest

from agents.vllm_backend import VLLMBackend
from env.distributed_clues import DistributedClues
from scripts.replay_distributed_clues import verify_batch, verify_diagnostic
from scripts.run_distributed_clues import PROMPT_VERSION, aggregate, run_episode


def response_for(text="{}"):
    return {
        "model": "local-qwen3-8b",
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def scripted_transport(request):
    obs = json.loads(request["messages"][0]["content"].split("\nObservation:\n", 1)[1])
    evidence = obs["known_evidence"]
    if obs["phase"] == "submit":
        candidates = set(p["id"] for p in obs["plans"])
        for clue in evidence:
            candidates.intersection_update(clue["allowed_plans"])
        action = {"action": "submit_report", "params": {
            "plan_id": sorted(candidates)[0], "evidence_ids": [c["id"] for c in evidence]}}
    else:
        action = {"action": "share", "params": {
            "evidence_ids": [c["id"] for c in evidence], "message": "Evidence attached."}}
    return response_for(json.dumps(action))


@pytest.mark.parametrize("bad_case", ["unknown", "duplicate", "not_list", "bad_item", "bad_plan", "wrong_action"])
def test_final_report_rejects_malformed_or_unreceived_evidence(bad_case):
    env = DistributedClues(7, "no_communication", rounds=1)
    for _ in range(3):
        env.step(env.current_agent, {"action": "wait", "params": {}})
    actor = env.current_agent
    known = [c["id"] for c in env.observe(actor)["known_evidence"]]
    action = {"action": "submit_report", "params": {"plan_id": env.audit()["answer"], "evidence_ids": known}}
    if bad_case == "unknown":
        action["params"]["evidence_ids"] = [next(e for e in env.audit()["clues"] if e not in known)]
    elif bad_case == "duplicate":
        action["params"]["evidence_ids"] = known * 2
    elif bad_case == "not_list":
        action["params"]["evidence_ids"] = known[0]
    elif bad_case == "bad_item":
        action["params"]["evidence_ids"] = [None]
    elif bad_case == "bad_plan":
        action["params"]["plan_id"] = []
    else:
        action = {"action": "wait", "params": {}}
    cursor = env.cursor
    before = copy.deepcopy(env._known)
    assert not env.step(actor, action)["valid"]
    assert env.cursor == cursor + 1
    assert env._known == before
    assert actor not in env.submissions


def test_aggregate_keeps_missing_submissions_in_denominator_and_pairs_only_matches():
    def row(seed, condition, correct, supported, invalid):
        return {"seed": seed, "replicate": 0, "condition": condition, "agents": 3,
                "correct_answers": correct, "supported_submissions": supported,
                "team_success": supported > 0, "invalid_actions": invalid,
                "turns_executed": 12, "tokens_in": 100, "tokens_out": 10}
    on = row(7, "communication", 2, 2, 1)
    off = row(7, "no_communication", 1, 0, 2)
    unmatched = row(8, "communication", 0, 0, 3)
    result = aggregate([on, off, unmatched])
    assert result["conditions"]["communication"]["answer_accuracy"] == pytest.approx(2 / 6)
    assert result["conditions"]["communication"]["supported_fraction"] == pytest.approx(2 / 6)
    assert result["conditions"]["communication"]["invalid_actions"] == 4
    assert result["conditions"]["communication"]["task_instances"] == 2
    assert len(result["paired_differences"]) == 1
    assert result["mean_paired_accuracy_difference"] == pytest.approx(1 / 3)
    assert aggregate([on, unmatched])["mean_paired_accuracy_difference"] is None
    assert aggregate([])["conditions"] == {}


@pytest.mark.parametrize("field", ["request_sha256", "identity"])
def test_vllm_rejects_corrupted_cached_receipt(tmp_path, field):
    calls = []
    def transport(request):
        calls.append(request)
        return response_for()
    backend = VLLMBackend("local-qwen3-8b", model_revision="test-revision", receipt_dir=tmp_path, transport=transport)
    backend.generate("prompt", 32, {}, sampling_seed=3)
    path = next(tmp_path.glob("*.json"))
    receipt = json.loads(path.read_text())
    if field == "identity":
        receipt[field]["request"]["temperature"] = 0.7
    else:
        receipt[field] = "incorrect"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity mismatch"):
        backend.generate("prompt", 32, {}, sampling_seed=3)
    assert len(calls) == 1


@pytest.mark.parametrize("bad_case", ["model", "negative_usage", "boolean_usage", "missing_usage", "empty_text"])
def test_vllm_rejects_invalid_response_and_retains_receipt(tmp_path, bad_case):
    response = response_for()
    if bad_case == "model":
        response["model"] = "different-model"
    elif bad_case == "negative_usage":
        response["usage"]["prompt_tokens"] = -1
    elif bad_case == "boolean_usage":
        response["usage"]["completion_tokens"] = True
    elif bad_case == "missing_usage":
        del response["usage"]["completion_tokens"]
    else:
        response["choices"][0]["message"]["content"] = " "
    backend = VLLMBackend("local-qwen3-8b", model_revision="test-revision", receipt_dir=tmp_path,
                          transport=lambda _: response)
    with pytest.raises(RuntimeError):
        backend.generate("prompt", 32, {})
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["response"] == response


@pytest.mark.parametrize("error", [HTTPError("http://127.0.0.1", 503, "Unavailable", {}, None), URLError("offline"), TimeoutError()])
def test_vllm_transport_errors_fail_without_fabricated_receipt(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("agents.vllm_backend.urlopen", fail)
    backend = VLLMBackend("local-qwen3-8b", model_revision="test-revision", receipt_dir=tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        backend.generate("prompt", 32, {})
    assert not list(tmp_path.glob("*.json"))


@pytest.fixture
def complete_batch(tmp_path):
    backend = VLLMBackend("local-qwen3-8b", model_revision="test-revision", receipt_dir=tmp_path / "receipts",
                          transport=scripted_transport)
    config = {"seeds": [7], "replicates": 1, "conditions": ["communication", "no_communication", "all_clues_single"], "rounds": 1}
    rows = [run_episode(tmp_path / f"seed-7-rep-0-{condition}", backend, seed=7, condition=condition, rounds=1, replicate=0)
            for condition in config["conditions"]]
    manifest = {"status": "complete", "completed_episodes": 3, "config": config,
                "model_revision": "test-revision", "prompt_version": PROMPT_VERSION,
                "temperature": 0.2, "top_p": 0.9, "max_new_tokens": 256}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "comparison.json").write_text(json.dumps(aggregate(rows)))
    return tmp_path


def test_complete_batch_reconciles_all_conditions_and_receipts(complete_batch):
    result = verify_batch(complete_batch)
    assert result["verified_episodes"] == 3
    assert result["verified_actions_and_receipts"] == 14


@pytest.mark.parametrize("corruption", ["temperature", "top_p", "max_tokens", "model", "finish_reason", "ledger", "aggregate"])
def test_complete_batch_rejects_artifact_corruption(complete_batch, corruption):
    if corruption in ("ledger", "aggregate"):
        path = complete_batch / ("episodes.jsonl" if corruption == "ledger" else "comparison.json")
        if corruption == "ledger":
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["tokens_in"] += 1
            path.write_text("\n".join(json.dumps(row) for row in rows))
        else:
            value = json.loads(path.read_text())
            value["conditions"]["communication"]["answer_accuracy"] = 0
            path.write_text(json.dumps(value))
    else:
        path = next((complete_batch / "receipts").glob("*.json"))
        receipt = json.loads(path.read_text())
        if corruption in ("temperature", "top_p", "max_tokens"):
            receipt["identity"]["request"][corruption] = 0.7 if corruption != "max_tokens" else 128
            # A consistent hash still must not bypass the recorded run settings.
            digest = hashlib.sha256(json.dumps(receipt["identity"], sort_keys=True).encode()).hexdigest()
            receipt["request_sha256"] = digest
            path.unlink()
            path = path.with_name(digest + ".json")
        elif corruption == "model":
            receipt["response"]["model"] = "different-model"
        else:
            receipt["response"]["choices"][0]["finish_reason"] = "length"
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_batch(complete_batch)


def test_diagnostic_cli_failure_preserves_incomplete_manifest(tmp_path, monkeypatch):
    from scripts import run_single_agent_diagnostic as diagnostic
    def fail(request):
        raise RuntimeError("offline test transport failed")
    def backend_factory(*args, **kwargs):
        return VLLMBackend(*args, **kwargs, transport=fail)
    monkeypatch.setattr(diagnostic, "VLLMBackend", backend_factory)
    out = tmp_path / "diagnostic"
    monkeypatch.setattr("sys.argv", ["diagnostic", "--output", str(out), "--seeds", "7", "--modes", "phase-v2"])
    with pytest.raises(RuntimeError, match="offline test"):
        diagnostic.main()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["completed_cases"] == 0
    assert not (out / "summary.json").exists()


def test_diagnostic_cli_success_and_regrading_rejects_changed_score(tmp_path, monkeypatch):
    from scripts import run_single_agent_diagnostic as diagnostic
    def backend_factory(*args, **kwargs):
        return VLLMBackend(*args, **kwargs, transport=scripted_transport)
    monkeypatch.setattr(diagnostic, "VLLMBackend", backend_factory)
    out = tmp_path / "diagnostic"
    monkeypatch.setattr("sys.argv", ["diagnostic", "--output", str(out), "--seeds", "7", "8", "--modes", "phase-v2"])
    diagnostic.main()
    assert verify_diagnostic(out)["verified_model_decisions_and_receipts"] == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["phase-v2"]["supported_correct"] == 2
    summary["phase-v2"]["supported_correct"] = 1
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary mismatch"):
        verify_diagnostic(out)

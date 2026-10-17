import copy
import itertools
import json

import pytest

from env.distributed_clues import DistributedClues
from scripts.run_distributed_clues import aggregate, policy_prompt, run_episode
from agents.language_backend import GenerationResult


def share_all(env):
    observation = env.observe(env.current_agent)
    return {"action": "share", "params": {
        "evidence_ids": [c["id"] for c in observation["known_evidence"]], "message": "Here is my evidence."}}


def solve(observation):
    evidence = observation["known_evidence"]
    choices = set(p["id"] for p in observation["plans"])
    for clue in evidence:
        choices.intersection_update(clue["allowed_plans"])
    return {"action": "submit_report", "params": {
        "plan_id": sorted(choices)[0], "evidence_ids": [e["id"] for e in evidence]}}


def finish(env):
    while not env.done:
        obs = env.observe(env.current_agent)
        action = solve(obs) if obs["phase"] == "submit" else share_all(env)
        env.step(env.current_agent, action)
    return env.summary()


def test_information_is_complementary_over_500_instances():
    answers, owners = set(), set()
    for seed in range(500):
        env = DistributedClues(seed)
        audit = env.audit()
        clues = list(audit["clues"].values())
        for size in (1, 2, 3):
            for subset in itertools.combinations(clues, size):
                intersection = set.intersection(*(set(c["allowed_plans"]) for c in subset))
                assert len(intersection) == 2 ** (3 - size)
                assert audit["answer"] in intersection
        assert all(len(v) == 1 for v in audit["initial_evidence"].values())
        answers.add(audit["answer"])
        owners.add(tuple(audit["owners"].values()))
    assert len(answers) == 8
    assert len(owners) == 6


def test_communication_controls_have_identical_private_instances():
    on, off = DistributedClues(7), DistributedClues(7, "no_communication")
    assert on.audit() == off.audit()
    assert on.schedule == off.schedule
    on_summary, off_summary = finish(on), finish(off)
    assert on_summary["supported_submissions"] == 3
    assert on_summary["new_evidence_deliveries"] == 6
    assert off_summary["supported_submissions"] == 0
    assert off_summary["new_evidence_deliveries"] == 0
    assert off_summary["evidence_coverage"] == {a: 1 / 3 for a in off.agents}


def test_all_clues_solvability_reference():
    env = DistributedClues(8, "all_clues_single")
    assert len(env.agents) == 1
    assert finish(env)["team_success"]


def test_observation_only_contains_own_or_received_evidence_and_is_detached():
    env = DistributedClues(2)
    agent = env.current_agent
    observation = env.observe(agent)
    known = {e["id"] for e in observation["known_evidence"]}
    assert known == set(env.audit()["initial_evidence"][agent])
    hidden = set(env.audit()["clues"]) - known
    assert all(e not in json.dumps(observation) for e in hidden)
    observation["known_evidence"][0]["allowed_plans"].clear()
    assert len(env.observe(agent)["known_evidence"][0]["allowed_plans"]) == 4


def test_cannot_fetch_or_share_other_agents_evidence():
    env = DistributedClues(3)
    actor = env.current_agent
    hidden = next(e for e in env.audit()["clues"] if e not in env.audit()["initial_evidence"][actor])
    result = env.step(actor, {"action": "share", "params": {"evidence_ids": [hidden], "message": ""}})
    assert not result["valid"]
    result = env.step(env.current_agent, {"action": "research", "params": {"doc_id": hidden}})
    assert not result["valid"]
    assert all(len(v) == 1 for v in env._known.values())


def test_no_communication_blocks_text_evidence_and_location_side_channel():
    left, right = DistributedClues(4, "no_communication"), DistributedClues(4, "no_communication")
    first = left.current_agent
    left.step(first, {"action": "move", "params": {"destination": "library"}})
    right.step(first, share_all(right))
    assert left.observe(left.current_agent) == right.observe(right.current_agent)
    assert not right._inbox[right.current_agent]


def test_sharing_requires_colocation():
    env = DistributedClues(4, rounds=2)
    first = env.current_agent
    env.step(first, {"action": "move", "params": {"destination": "library"}})
    result = env.step(env.current_agent, share_all(env))
    assert first not in result["delivered_to"]
    assert len(env._known[first]) == 1


def test_final_report_grades_answer_and_received_evidence_without_feedback_leak():
    env = DistributedClues(7, rounds=1)
    for _ in range(3):
        env.step(env.current_agent, share_all(env))
    actor = env.current_agent
    action = solve(env.observe(actor))
    result = env.step(actor, action)
    assert result["supported_success"]
    next_observation = env.observe(env.current_agent)
    assert "submissions" not in next_observation
    assert "answer_correct" not in json.dumps(next_observation)
    # A valid but incorrect candidate is not credited even with all evidence.
    wrong = copy.deepcopy(action)
    wrong["params"]["plan_id"] = next(p["id"] for p in env.plans if p["id"] != env.audit()["answer"])
    assert not env.step(env.current_agent, wrong)["supported_success"]


@pytest.mark.parametrize("action", [
    None, [], {"action": "wait", "params": [], "extra": 1},
    {"action": "move", "params": {"destination": []}},
    {"action": "share", "params": {"evidence_ids": [None], "message": ""}},
    {"action": "share", "params": {"evidence_ids": ["x", "x"], "message": ""}},
    {"action": "share", "params": {"evidence_ids": [], "message": "x" * 281}},
    {"action": "submit_report", "params": {"plan_id": "plan_1", "evidence_ids": []}},
])
def test_bad_action_consumes_turn_without_fallback(action):
    env = DistributedClues()
    before = copy.deepcopy(env._known)
    assert not env.step(env.current_agent, action)["valid"]
    assert env.cursor == 1
    assert env._known == before


def test_out_of_turn_and_after_done_rejected():
    env = DistributedClues()
    wrong = next(a for a in env.agents if a != env.current_agent)
    with pytest.raises(ValueError):
        env.observe(wrong)
    with pytest.raises(ValueError):
        env.step(wrong, {"action": "wait", "params": {}})
    finish(env)
    with pytest.raises(ValueError):
        env.step(env.agents[0], {"action": "wait", "params": {}})


def test_runner_records_real_policy_outputs_and_separates_replicates(tmp_path):
    class ScriptedPolicy:
        def __init__(self):
            self.seeds = []

        def generate(self, prompt, max_new_tokens, alphas, *, sampling_seed):
            self.seeds.append(sampling_seed)
            assert alphas == {}
            obs = json.loads(prompt.split("\nObservation:\n", 1)[1])
            if obs["phase"] == "submit":
                action = solve(obs)
            else:
                action = {"action": "share", "params": {
                    "evidence_ids": [c["id"] for c in obs["known_evidence"]], "message": ""}}
            return GenerationResult(text=json.dumps(action), tokens_in=100, tokens_out=50)

    backend = ScriptedPolicy()
    first = run_episode(tmp_path / "one", backend, seed=7, condition="communication", rounds=2, replicate=0)
    seeds0 = backend.seeds.copy()
    backend.seeds.clear()
    second = run_episode(tmp_path / "two", backend, seed=7, condition="communication", rounds=2, replicate=1)
    assert seeds0 != backend.seeds
    assert first["team_success"] and second["team_success"]
    traces = [json.loads(line) for line in (tmp_path / "one" / "trace.jsonl").read_text().splitlines()]
    assert len(traces) == 9
    assert first["tokens_out"] == 450
    assert aggregate([first, second])["conditions"]["communication"]["task_instances"] == 1
    from scripts.replay_distributed_clues import verify_episode
    assert verify_episode(tmp_path / "one", seed=7, condition="communication", rounds=2) == 9
    import hashlib
    original_prompt = traces[0]["prompt"]
    traces[0]["prompt"] = "Privileged answer: plan_4. " + original_prompt
    traces[0]["prompt_sha256"] = hashlib.sha256(traces[0]["prompt"].encode()).hexdigest()
    (tmp_path / "one" / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in traces))
    with pytest.raises(ValueError, match="Full policy prompt mismatch"):
        verify_episode(tmp_path / "one", seed=7, condition="communication", rounds=2)
    traces[0]["prompt"] = original_prompt
    traces[0]["prompt_sha256"] = hashlib.sha256(original_prompt.encode()).hexdigest()
    traces[0]["observation"]["known_evidence"].clear()
    (tmp_path / "one" / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in traces))
    with pytest.raises(ValueError, match="Observation mismatch"):
        verify_episode(tmp_path / "one", seed=7, condition="communication", rounds=2)


def test_no_communication_correct_guess_is_not_supported_success():
    env = DistributedClues(1, "no_communication", rounds=1)
    for _ in range(3):
        env.step(env.current_agent, {"action": "wait", "params": {}})
    actor = env.current_agent
    result = env.step(actor, {"action": "submit_report", "params": {
        "plan_id": env.audit()["answer"], "evidence_ids": env.audit()["initial_evidence"][actor]}})
    assert result["answer_correct"]
    assert not result["supported_success"]


def test_final_prompt_exposes_only_legal_action_schema():
    env = DistributedClues(rounds=1)
    for _ in range(3):
        env.step(env.current_agent, {"action": "wait", "params": {}})
    prompt = policy_prompt(env.observe(env.current_agent))
    assert prompt.startswith("FINAL TURN")
    assert '"action":"submit_report"' in prompt
    assert '"action":"share"' not in prompt


def test_single_agent_diagnostic_counts_invalid_model_output():
    from scripts.run_single_agent_diagnostic import run_case

    class InvalidPolicy:
        def generate(self, prompt, max_new_tokens, alphas, *, sampling_seed):
            assert "ALL THREE lists" in prompt
            assert "attendee availability" in prompt
            return GenerationResult(text="not JSON", tokens_in=10, tokens_out=2)

    result = run_case(InvalidPolicy(), 7, 0, "intersection-v1")
    assert not result["result"]["valid"]
    assert result["action"] is None
    assert result["fixture_wait_turns"] == 1
    assert result["model_calls"] == 1
    assert len(result["observation"]["known_evidence"]) == 3

import json
from unittest.mock import patch

import pytest

from agents.anthropic_backend import AnthropicBackend
from agents.vllm_backend import VLLMBackend
from agents.agent import Agent
from agents.planner import PlanSuggestion
from evaluation.personality import strict_json, validate_persona
from orchestrator.cli import build_agents, build_language_backend, load_config
from agents.language_backend import MockBackend
from env.world import World
from safety.governor import SafetyConfig, SafetyGovernor
from pathlib import Path


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{} extra'])
def test_strict_json_rejects_ambiguous_values(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_hosted_receipts_replay_without_another_call(tmp_path):
    response = {"model": "test-model-version", "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "content": [{"type": "text", "text": "hello"}]}
    with patch.object(AnthropicBackend, "_post", return_value=response) as transport:
        backend = AnthropicBackend("test-model", receipt_dir=tmp_path)
        first = backend.generate("prompt", 32, {}, sampling_seed=2)
        second = backend.generate("prompt", 32, {}, sampling_seed=2)
        assert first == second
        assert transport.call_count == 1
        assert first.sampling_seed is None
        assert first.input_ids == [] and not first.steering_applied
        backend.generate("prompt", 32, {}, sampling_seed=3)
        assert transport.call_count == 2
        with pytest.raises(ValueError, match="activation"):
            backend.generate("prompt", 32, {"A": 0.5})


def test_hosted_truncation_is_not_a_success(tmp_path):
    backend = AnthropicBackend("test", receipt_dir=tmp_path,
                               transport=lambda _: {"stop_reason": "max_tokens"})
    with pytest.raises(RuntimeError, match="incomplete"):
        backend.generate("prompt", 32, {})
    assert not list(tmp_path.glob("*.json"))


def test_prompt_persona_changes_do_not_change_initial_roles_locations():
    config = load_config(Path("configs/run.local-8b.yaml"))
    def build(config):
        return build_agents("test", config, World(), MockBackend(),
                            SafetyGovernor(SafetyConfig()), "research", suppress_alphas=True)
    baseline = build(config)
    config["prompt_persona"] = {"coefficients": {"A": 0.8}, "jitter": 0.1}
    treatment = build(config)
    assert [(a.state.role, a.state.location_id, a.state.goals) for a in baseline] == [
        (a.state.role, a.state.location_id, a.state.goals) for a in treatment]
    assert all(a.persona_alphas()["A"] == 0 for a in treatment)
    assert all(a._prompt_persona["A"] > 0.6 for a in treatment)


def test_strict_action_mode_records_fallback():
    agent = object.__new__(Agent)
    agent._structured_actions_enabled = True
    agent._strict_action_json = True
    suggestion = PlanSuggestion("talk", {"utterance": "hello"}, "hello")
    good = json.dumps({"action": "talk", "params": {}, "utterance": "hi"})
    assert agent._parse_structured_action(good, suggestion)[3] == "model"
    result = agent._parse_structured_action(good + " extra", suggestion)
    assert result[3] == "planner_fallback" and result[4]


@pytest.mark.parametrize("values", [{"A": float("nan")}, {"Z": 1}, {"A": True}, {"A": 2}])
def test_invalid_persona_rejected(values):
    with pytest.raises(ValueError):
        validate_persona(values)


def test_hosted_backend_rejects_steering_before_loading_model():
    with pytest.raises(ValueError, match="steering.enabled"):
        build_language_backend({"inference": {"backend": "anthropic"}}, {}, {}, False)


def test_vllm_records_and_replays_actual_response(tmp_path):
    calls = []
    def transport(request):
        calls.append(request)
        return {"model": "local-test", "choices": [{"finish_reason": "stop",
                "message": {"content": '{"action":"talk","params":{},"utterance":"hi"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    backend = VLLMBackend("local-test", model_revision="revision", receipt_dir=tmp_path,
                          transport=transport)
    result = backend.generate("prompt", 128, {}, sampling_seed=42)
    assert backend.generate("prompt", 128, {}, sampling_seed=42) == result
    assert len(calls) == 1 and calls[0]["seed"] == 42
    assert calls[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert result.tokens_in == 100 and not result.input_ids
    assert result.sampling_seed == 42 and not result.steering_applied
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_vllm_refuses_external_endpoint(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        VLLMBackend("test", model_revision="r", receipt_dir=tmp_path,
                    base_url="https://api.example.com/v1")


def test_vllm_preserves_truncated_response_without_calling_it_success(tmp_path):
    backend = VLLMBackend("test", model_revision="r", receipt_dir=tmp_path,
                          transport=lambda _: {"choices": [{"finish_reason": "length"}]})
    with pytest.raises(RuntimeError, match="incomplete"):
        backend.generate("prompt", 32, {})
    assert len(list(tmp_path.glob("*.json"))) == 1

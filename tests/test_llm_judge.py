import pytest

from steering.llm_judge import (
    LikertParseError,
    ModelFamilyError,
    StaticJudgeClient,
    enforce_distinct_model_families,
    extract_model_family,
    parse_likert_scores,
    score_text_with_judge,
)


def test_extract_model_family_handles_common_model_names() -> None:
    assert extract_model_family("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert extract_model_family("Qwen/Qwen2.5-32B-Instruct") == "qwen"
    assert extract_model_family("anthropic/claude-3-5-sonnet") == "claude"
    assert extract_model_family("gpt-4o-mini") == "gpt"


def test_enforce_distinct_model_families_rejects_same_family() -> None:
    with pytest.raises(ModelFamilyError):
        enforce_distinct_model_families(
            "meta-llama/Llama-3.1-8B-Instruct",
            "llama-3.3-70b",
        )

    pair = enforce_distinct_model_families(
        "meta-llama/Llama-3.1-8B-Instruct",
        "gpt-4o-mini",
    )
    assert pair.target_family == "llama"
    assert pair.judge_family == "gpt"


def test_parse_likert_scores_from_json_and_text() -> None:
    assert parse_likert_scores(
        '{"scores": {"extraversion": 5, "agreeableness": "4/5"}}',
        ["E", "A"],
    ) == {"E": 5, "A": 4}

    parsed = parse_likert_scores(
        "Extraversion: 2\nAgreeableness=3\nConscientiousness - 4",
        ["E", "A", "C"],
    )
    assert parsed == {"E": 2, "A": 3, "C": 4}


def test_parse_likert_scores_rejects_missing_expected_trait() -> None:
    with pytest.raises(LikertParseError):
        parse_likert_scores("Extraversion: 4", ["E", "A"])


def test_static_judge_client_can_be_mocked_through_score_path() -> None:
    client = StaticJudgeClient(['{"E": 5, "A": 4}'])

    result = score_text_with_judge(
        client,
        "I gathered everyone, welcomed the newcomer, and kept the group moving.",
        judge_model="gpt-4o-mini",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
        traits=["E", "A"],
    )

    assert result.scores == {"E": 5, "A": 4}
    assert len(client.requests) == 1
    assert "Generated text" in client.requests[0].prompt

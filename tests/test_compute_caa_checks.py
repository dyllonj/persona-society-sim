import numpy as np
import pytest

from steering.compute_caa import (
    _validate_answer_token_mask,
    directional_agreement,
    enforce_orthogonality,
    resolve_caa_config,
    separability_d_prime,
    validate_model_config,
)


class DummyTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def convert_ids_to_tokens(self, token_ids):
        return [self.mapping[token_id] for token_id in token_ids]


def test_validate_answer_token_mask_allows_unmasked_letter():
    tokenizer = DummyTokenizer({1: "A", 2: "B"})
    _validate_answer_token_mask(tokenizer, [1, 2], 0, "A", [1, 1])


def test_validate_answer_token_mask_rejects_mismatch():
    tokenizer = DummyTokenizer({1: "B"})
    with pytest.raises(ValueError):
        _validate_answer_token_mask(tokenizer, [1], 0, "A", [1])


def test_validate_answer_token_mask_rejects_masked_token():
    tokenizer = DummyTokenizer({1: "A"})
    with pytest.raises(ValueError):
        _validate_answer_token_mask(tokenizer, [1], 0, "A", [0])


def test_enforce_orthogonality_blocks_overlapping_layers():
    candidate = {0: np.array([1.0, 0.0], dtype=np.float32)}
    existing = [("Agreeableness", 0, np.array([0.9, 0.0], dtype=np.float32))]
    with pytest.raises(ValueError):
        enforce_orthogonality(candidate, existing, threshold=0.1)


def test_enforce_orthogonality_allows_orthogonal_layers():
    candidate = {0: np.array([1.0, 0.0], dtype=np.float32)}
    existing = [("Openness", 0, np.array([0.0, 1.0], dtype=np.float32))]
    enforce_orthogonality(candidate, existing, threshold=0.1)


def test_directional_agreement_scores_diff_alignment():
    diffs = np.array([[1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]], dtype=np.float32)

    assert directional_agreement(diffs) == pytest.approx(1 / 3)


def test_separability_d_prime_uses_projection_variance():
    diffs = np.array([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    assert separability_d_prime(diffs) > 1.0


def test_resolve_caa_config_uses_trait_yaml_metadata(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "extraversion.jsonl"
    prompt_file.write_text("", encoding="utf-8")
    config_path = config_dir / "steering.layers.yaml"
    config_path.write_text(
        """
vector_root: ../vectors
defaults:
  model: test/model
  layers: [1, 2]
  prompt_dir: ../prompts
  num_hidden_layers: 4
traits:
  E:
    name: extraversion
    vector_store_id: E_cfg
    prompt_file: ../prompts/extraversion.jsonl
    layers: [2, 3]
""",
        encoding="utf-8",
    )

    resolved = resolve_caa_config("E", config_path=config_path)

    assert resolved.prompt_file == prompt_file.resolve()
    assert resolved.output_dir == (tmp_path / "vectors").resolve()
    assert resolved.model == "test/model"
    assert resolved.layers == (2, 3)
    assert resolved.vector_store_id == "E_cfg"
    assert resolved.expected_num_hidden_layers == 4


def test_resolve_caa_config_keeps_explicit_cli_values_usable(tmp_path):
    config_path = tmp_path / "steering.layers.yaml"
    config_path.write_text(
        """
vector_root: vectors
defaults:
  model: config/model
  layers: [1]
  num_hidden_layers: 4
traits:
  E:
    vector_store_id: E_cfg
    prompt_file: prompts/e.jsonl
    layers: [2]
""",
        encoding="utf-8",
    )
    prompt_file = tmp_path / "explicit.jsonl"
    output_dir = tmp_path / "explicit-vectors"

    resolved = resolve_caa_config(
        "E",
        config_path=config_path,
        prompt_file=prompt_file,
        output_dir=output_dir,
        model="explicit/model",
        layers=[0, 3, 3],
        vector_store_id="explicit-store",
    )

    assert resolved.prompt_file == prompt_file
    assert resolved.output_dir == output_dir
    assert resolved.model == "explicit/model"
    assert resolved.layers == (0, 3)
    assert resolved.vector_store_id == "explicit-store"
    assert resolved.expected_num_hidden_layers is None


class DummyModel:
    class config:
        num_hidden_layers = 4


def test_validate_model_config_rejects_yaml_model_layer_mismatch(tmp_path):
    resolved = resolve_caa_config(
        "E",
        config_path=_write_minimal_config(tmp_path, num_hidden_layers=8),
    )

    with pytest.raises(ValueError, match="defaults.num_hidden_layers=8"):
        validate_model_config(DummyModel(), resolved)


def test_validate_model_config_rejects_out_of_range_layer(tmp_path):
    config_path = _write_minimal_config(tmp_path, num_hidden_layers=4, layers=[4])
    resolved = resolve_caa_config("E", config_path=config_path)

    with pytest.raises(ValueError, match="valid decoder layer ids are 0..3"):
        validate_model_config(DummyModel(), resolved)


def _write_minimal_config(
    tmp_path,
    *,
    num_hidden_layers: int,
    layers: list[int] | None = None,
):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    (prompt_dir / "e.jsonl").write_text("", encoding="utf-8")
    config_path = tmp_path / "steering.layers.yaml"
    layer_values = layers or [1, 2]
    config_path.write_text(
        f"""
vector_root: vectors
defaults:
  model: test/model
  layers: [1]
  num_hidden_layers: {num_hidden_layers}
traits:
  E:
    prompt_file: prompts/e.jsonl
    layers: {layer_values}
""",
        encoding="utf-8",
    )
    return config_path

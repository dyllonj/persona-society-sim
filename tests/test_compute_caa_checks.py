import numpy as np
import pytest

from steering.compute_caa import enforce_orthogonality, _validate_answer_token_mask


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

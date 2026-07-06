from pathlib import Path

import numpy as np
import pytest

from orchestrator.cli import load_trait_vectors
from steering.vector_store import VectorStore, coerce_polarity


def test_vector_store_calibrated_vectors_apply_polarity(tmp_path: Path):
    store = VectorStore(tmp_path)
    store.save_vectors(
        "E",
        "caa_ab",
        {36: np.array([1.0, -2.0], dtype=np.float32)},
        model_name="mock-model",
        train_set_hash="hash",
        norms={36: 2.0},
        hyperparameters={},
        num_train_prompts=1,
        vector_store_id="E_store",
        polarity=-1.0,
    )

    bundle = store.load("E_store")

    assert bundle.polarity == -1.0
    np.testing.assert_allclose(bundle.vectors[36], np.array([1.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(
        bundle.calibrated_vectors()[36], np.array([-1.0, 2.0], dtype=np.float32)
    )


def test_orchestrator_load_trait_vectors_applies_config_polarity(tmp_path: Path):
    store = VectorStore(tmp_path)
    store.save_vectors(
        "E",
        "caa_ab",
        {36: np.array([1.0, -2.0], dtype=np.float32)},
        model_name="mock-model",
        train_set_hash="hash",
        norms={36: 2.0},
        hyperparameters={},
        num_train_prompts=1,
        vector_store_id="E_store",
    )

    vectors, _ = load_trait_vectors(
        ["E"],
        tmp_path,
        vector_metadata={
            "vector_root": str(tmp_path),
            "traits": {
                "E": {
                    "vector_store_id": "E_store",
                    "layers": [36],
                    "polarity": -1.0,
                }
            },
        },
    )

    np.testing.assert_allclose(vectors["E"][36], np.array([-1.0, 2.0], dtype=np.float32))


def test_coerce_polarity_rejects_magnitude_scaling():
    with pytest.raises(ValueError):
        coerce_polarity(0.5)

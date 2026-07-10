from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

jlens = pytest.importorskip("jlens")

from interpretability import trait_report  # noqa: E402
from interpretability.common import sha256_file, sha256_json  # noqa: E402


def _fixture(tmp_path: Path, *, bad_model: bool = False, bad_width: bool = False) -> dict[str, Path]:
    vectors = tmp_path / "vectors"
    vectors.mkdir(parents=True)
    np.save(vectors / "E.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(vectors / "A.npy", np.array([1.0, 1.0, 0.0] if bad_width else [1.0, 1.0], dtype=np.float32))
    index = vectors / "index.jsonl"
    entries = [
        {
            "vector_store_id": "E-test",
            "trait": "E",
            "method": "caa_ab",
            "layer_id": 0,
            "vector_path": str(vectors / "E.npy"),
            "train_set_hash": "etrain",
            "eval_set_hash": "eeval",
        },
        {
            "vector_store_id": "A-test",
            "trait": "A",
            "method": "caa_ab",
            "layer_id": 1,
            "vector_path": str(vectors / "A.npy"),
            "train_set_hash": "atrain",
            "eval_set_hash": "aeval",
        },
    ]
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    metadata = tmp_path / "steering.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "vector_root": "vectors",
                "defaults": {
                    "model": "wrong/model" if bad_model else "tiny/model",
                    "num_hidden_layers": 3,
                },
                "traits": {
                    "E": {
                        "name": "extraversion",
                        "vector_store_id": "E-test",
                        "layers": [0],
                        "polarity": -1.0,
                    },
                    "A": {
                        "name": "agreeableness",
                        "vector_store_id": "A-test",
                        "layers": [1],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    lens_path = tmp_path / "lens.pt"
    lens = jlens.JacobianLens(
        {
            0: torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
            1: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        n_prompts=5,
        d_model=2,
    )
    lens.save(str(lens_path), dtype=torch.float32)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "lens_id": "tiny-lens",
                "lens_sha256": sha256_file(lens_path),
                "model_id": "tiny/model",
                "model_revision": "abc123",
                "tokenizer_revision": "abc123",
                "model_config_sha256": sha256_json({"width": 2}),
                "jlens_git_commit": "test-commit",
                "jlens_version": "test",
                "dtype": "fp32",
                "d_model": 2,
                "n_layers": 3,
                "source_layers": [0, 1],
                "target_layer": 2,
                "n_prompts": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {"lens": lens_path, "manifest": manifest, "metadata": metadata}


def test_build_report_computes_transport_gains_and_full_matrices(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = trait_report.build_report(
        lens_path=paths["lens"],
        manifest_path=paths["manifest"],
        metadata_path=paths["metadata"],
        alpha_values=["E=0.5", "A=2.0"],
        skip_vocab=True,
        require_complete=True,
    )

    assert report["coverage"]["complete"] is True
    assert report["component_space"]["labels"] == ["A@1", "E@0"]
    assert len(report["component_space"]["gram"]) == 2
    assert all(len(row) == 2 for row in report["component_space"]["cosine"])

    by_label = {row["label"]: row for row in report["components"]}
    assert by_label["E@0"]["raw_norm"] == pytest.approx(1.0)
    assert by_label["E@0"]["transported_norm"] == pytest.approx(2.0)
    assert by_label["E@0"]["directional_gain"] == pytest.approx(2.0)
    assert by_label["E@0"]["isotropic_rms_gain"] == pytest.approx(math.sqrt(2.5))
    assert by_label["E@0"]["gain_ratio"] == pytest.approx(2.0 / math.sqrt(2.5))
    assert report["component_space"]["cosine"][0][1] == pytest.approx(-1 / math.sqrt(2))
    assert report["alphas"] == {"A": 2.0, "E": 0.5}
    assert report["alpha_weighted_combined_effect"]["norm"] == pytest.approx(math.sqrt(5.0))
    assert report["vocabulary_projections"]["status"] == "skipped"
    assert len(report["analysis_sha256"]) == 64


def test_report_rejects_lens_hash_model_and_vector_width_mismatches(tmp_path: Path) -> None:
    hash_paths = _fixture(tmp_path / "hash")
    manifest = json.loads(hash_paths["manifest"].read_text(encoding="utf-8"))
    manifest["lens_sha256"] = "0" * 64
    hash_paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="lens hash mismatch"):
        trait_report.build_report(
            lens_path=hash_paths["lens"],
            manifest_path=hash_paths["manifest"],
            metadata_path=hash_paths["metadata"],
        )

    model_paths = _fixture(tmp_path / "model", bad_model=True)
    with pytest.raises(ValueError, match="vector/lens model mismatch"):
        trait_report.build_report(
            lens_path=model_paths["lens"],
            manifest_path=model_paths["manifest"],
            metadata_path=model_paths["metadata"],
        )

    width_paths = _fixture(tmp_path / "width", bad_width=True)
    with pytest.raises(ValueError, match="vector shape mismatch"):
        trait_report.build_report(
            lens_path=width_paths["lens"],
            manifest_path=width_paths["manifest"],
            metadata_path=width_paths["metadata"],
        )


def test_cli_writes_json_and_markdown_without_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    prefix = tmp_path / "reports" / "trait-space"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trait_report",
            "--lens",
            str(paths["lens"]),
            "--lens-manifest",
            str(paths["manifest"]),
            "--vector-metadata",
            str(paths["metadata"]),
            "--output-prefix",
            str(prefix),
            "--skip-vocab",
            "--require-complete",
        ],
    )

    trait_report.main()

    payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert payload["coverage"]["complete"] is True
    assert "# Jacobian-space trait report" in markdown
    assert "Transported component cosine matrix" in markdown
    assert not list(prefix.parent.glob(".*.tmp.*"))

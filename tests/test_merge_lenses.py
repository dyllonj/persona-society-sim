from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from interpretability.common import sha256_file, sha256_json, write_json_atomic
from interpretability.merge_lenses import merge_lens_artifacts

jlens = pytest.importorskip("jlens")


def _make_artifact(
    root: Path,
    jacobians: dict[int, torch.Tensor],
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    corpus_path = root / "fit_prompts.jsonl"
    corpus_path.write_text('{"text":"held constant"}\n', encoding="utf-8")
    lens = jlens.JacobianLens(jacobians, n_prompts=3, d_model=2)
    lens_path = root / "lens.pt"
    lens.save(str(lens_path), dtype=torch.float16)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "lens_id": f"parent-{root.name}",
        "lens_sha256": sha256_file(lens_path),
        "jlens_git_commit": "abc123",
        "jlens_version": "0.1.0",
        "torch_version": "2.7.1",
        "transformers_version": "5.13.0",
        "model_id": "tiny/model",
        "model_revision": "model-commit",
        "tokenizer_revision": "tokenizer-commit",
        "model_config_sha256": sha256_json({"hidden_size": 2}),
        "dtype": "bf16",
        "quantization": None,
        "d_model": 2,
        "n_layers": 8,
        "source_layers": sorted(jacobians),
        "target_layer": 7,
        "corpus_name": "unit-corpus",
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_path": corpus_path.name,
        "n_prompts_requested": 3,
        "n_prompts": 3,
        "max_seq_len": 32,
        "skip_first": 4,
        "dim_batch": 2,
    }
    manifest.update(overrides or {})
    write_json_atomic(root / "manifest.json", manifest)
    return root


def test_merge_lens_artifacts_unions_layers_and_records_parent_hashes(tmp_path):
    first = _make_artifact(
        tmp_path / "first",
        {1: torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
    )
    second = _make_artifact(
        tmp_path / "second",
        {4: torch.tensor([[2.0, 0.0], [0.0, 2.0]])},
    )

    output = tmp_path / "union"
    manifest = merge_lens_artifacts([first, second], output)

    merged = jlens.JacobianLens.load(str(output / "lens.pt"))
    assert merged.source_layers == [1, 4]
    assert merged.n_prompts == 3
    assert torch.equal(merged.jacobians[1], torch.eye(2))
    assert torch.equal(merged.jacobians[4], torch.eye(2) * 2)
    assert manifest["lens_sha256"] == sha256_file(output / "lens.pt")
    assert sha256_file(output / "fit_prompts.jsonl") == manifest["corpus_sha256"]
    assert [
        parent["manifest_sha256"]
        for parent in manifest["merge_provenance"]["parents"]
    ] == [sha256_file(first / "manifest.json"), sha256_file(second / "manifest.json")]
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_merge_lens_artifacts_rejects_manifest_mismatch(tmp_path):
    first = _make_artifact(tmp_path / "first", {1: torch.eye(2)})
    second = _make_artifact(
        tmp_path / "second",
        {4: torch.eye(2)},
        overrides={"model_revision": "different-commit"},
    )

    with pytest.raises(ValueError, match="model_revision"):
        merge_lens_artifacts([first, second], tmp_path / "union")

    assert not (tmp_path / "union").exists()


def test_merge_lens_artifacts_rejects_unequal_overlap(tmp_path):
    first = _make_artifact(tmp_path / "first", {1: torch.eye(2)})
    second = _make_artifact(tmp_path / "second", {1: torch.eye(2) * 2})

    with pytest.raises(ValueError, match="overlapping layer 1 differs"):
        merge_lens_artifacts([first, second], tmp_path / "union")


def test_merge_lens_artifacts_accepts_exact_overlap(tmp_path):
    matrix = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    first = _make_artifact(tmp_path / "first", {1: matrix, 3: torch.eye(2)})
    second = _make_artifact(tmp_path / "second", {1: matrix, 4: torch.eye(2) * 4})

    output = tmp_path / "union"
    manifest = merge_lens_artifacts([first, second], output)

    assert manifest["source_layers"] == [1, 3, 4]
    assert manifest["merge_provenance"]["overlap_layers"] == [1]


def test_merge_lens_artifacts_rejects_tampered_lens(tmp_path):
    first = _make_artifact(tmp_path / "first", {1: torch.eye(2)})
    second = _make_artifact(tmp_path / "second", {4: torch.eye(2)})
    (second / "lens.pt").write_bytes(b"not the manifested lens")

    with pytest.raises(ValueError, match="lens hash mismatch"):
        merge_lens_artifacts([first, second], tmp_path / "union")

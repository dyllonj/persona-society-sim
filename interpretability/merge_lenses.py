from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:  # Supports both `python -m` and direct script execution.
    from .common import sha256_file, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from common import sha256_file, write_json_atomic  # type: ignore


# These fields determine whether independently fitted matrices estimate the same
# mapping. Artifact identity, source layers, timestamps, paths, and elapsed time
# are deliberately excluded.
COMPATIBILITY_FIELDS = (
    "schema_version",
    "jlens_git_commit",
    "jlens_version",
    "torch_version",
    "transformers_version",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "model_config_sha256",
    "dtype",
    "quantization",
    "d_model",
    "n_layers",
    "target_layer",
    "corpus_name",
    "corpus_sha256",
    "n_prompts_requested",
    "n_prompts",
    "max_seq_len",
    "skip_first",
    "dim_batch",
)


@dataclass(frozen=True)
class ParentArtifact:
    root: Path
    lens_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    lens_sha256: str
    jacobians: dict[int, torch.Tensor]
    serialization_dtype: torch.dtype


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge compatible Jacobian Lens artifacts over disjoint source layers"
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help="parent artifact directory containing lens.pt and manifest.json; repeat twice or more",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _require_manifest_fields(manifest: dict[str, Any], path: Path) -> None:
    required = {
        *COMPATIBILITY_FIELDS,
        "lens_id",
        "lens_sha256",
        "source_layers",
        "corpus_path",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")


def _safe_artifact_file(root: Path, relative_path: Any, *, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{root / 'manifest.json'} has invalid {field}")
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes artifact directory: {relative_path!r}")
    return candidate


def _load_parent(root: Path) -> ParentArtifact:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    lens_path = root / "lens.pt"
    if not root.is_dir():
        raise FileNotFoundError(f"parent artifact directory does not exist: {root}")
    if not manifest_path.is_file() or not lens_path.is_file():
        raise FileNotFoundError(f"{root} must contain manifest.json and lens.pt")

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    manifest: dict[str, Any] = manifest_payload
    _require_manifest_fields(manifest, manifest_path)

    actual_lens_sha256 = sha256_file(lens_path)
    if actual_lens_sha256 != manifest["lens_sha256"]:
        raise ValueError(
            f"lens hash mismatch for {root}: manifest={manifest['lens_sha256']}, "
            f"actual={actual_lens_sha256}"
        )

    corpus_path = _safe_artifact_file(root, manifest["corpus_path"], field="corpus_path")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"manifested corpus does not exist: {corpus_path}")
    actual_corpus_sha256 = sha256_file(corpus_path)
    if actual_corpus_sha256 != manifest["corpus_sha256"]:
        raise ValueError(
            f"corpus hash mismatch for {root}: manifest={manifest['corpus_sha256']}, "
            f"actual={actual_corpus_sha256}"
        )

    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "J" not in checkpoint:
        raise ValueError(f"{lens_path} is not a JacobianLens artifact")
    raw_jacobians = checkpoint["J"]
    if not isinstance(raw_jacobians, dict) or not raw_jacobians:
        raise ValueError(f"{lens_path} has no Jacobian matrices")
    valid_entries = all(
        isinstance(layer, int) and isinstance(matrix, torch.Tensor)
        for layer, matrix in raw_jacobians.items()
    )
    if not valid_entries:
        raise ValueError(f"{lens_path} has invalid Jacobian entries")

    source_layers = sorted(raw_jacobians)
    if manifest["source_layers"] != source_layers:
        raise ValueError(
            f"source_layers mismatch for {root}: manifest={manifest['source_layers']}, "
            f"lens={source_layers}"
        )
    if checkpoint.get("source_layers") != source_layers:
        raise ValueError(f"checkpoint source_layers disagree with its Jacobian keys: {lens_path}")
    if checkpoint.get("d_model") != manifest["d_model"]:
        raise ValueError(f"d_model mismatch between lens and manifest: {root}")
    if checkpoint.get("n_prompts") != manifest["n_prompts"]:
        raise ValueError(f"n_prompts mismatch between lens and manifest: {root}")

    d_model = manifest["d_model"]
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError(f"invalid d_model in {manifest_path}: {d_model!r}")
    expected_shape = (d_model, d_model)
    for layer, matrix in raw_jacobians.items():
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"layer {layer} in {lens_path} has shape {tuple(matrix.shape)}, "
                f"expected {expected_shape}"
            )
        if not matrix.is_floating_point():
            raise ValueError(f"layer {layer} in {lens_path} is not floating point")

    dtypes = {matrix.dtype for matrix in raw_jacobians.values()}
    if len(dtypes) != 1:
        raise ValueError(f"{lens_path} mixes Jacobian serialization dtypes: {dtypes}")

    return ParentArtifact(
        root=root,
        lens_path=lens_path,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        lens_sha256=actual_lens_sha256,
        jacobians=raw_jacobians,
        serialization_dtype=next(iter(dtypes)),
    )


def _validate_compatibility(parents: list[ParentArtifact]) -> None:
    reference = parents[0]
    for parent in parents[1:]:
        differences = {
            field: (reference.manifest[field], parent.manifest[field])
            for field in COMPATIBILITY_FIELDS
            if reference.manifest[field] != parent.manifest[field]
        }
        if differences:
            formatted = ", ".join(
                f"{field}: {left!r} != {right!r}"
                for field, (left, right) in differences.items()
            )
            raise ValueError(
                f"incompatible lens manifests {reference.root} and {parent.root}: {formatted}"
            )
        if parent.serialization_dtype != reference.serialization_dtype:
            raise ValueError(
                "incompatible lens serialization dtypes: "
                f"{reference.root}={reference.serialization_dtype}, "
                f"{parent.root}={parent.serialization_dtype}"
            )


def _union_jacobians(
    parents: list[ParentArtifact],
) -> tuple[dict[int, torch.Tensor], list[int]]:
    union: dict[int, torch.Tensor] = {}
    overlaps: list[int] = []
    owners: dict[int, Path] = {}
    for parent in parents:
        for layer, matrix in parent.jacobians.items():
            if layer not in union:
                union[layer] = matrix
                owners[layer] = parent.root
                continue
            previous = union[layer]
            if previous.dtype != matrix.dtype or not torch.equal(previous, matrix):
                raise ValueError(
                    f"overlapping layer {layer} differs between {owners[layer]} and "
                    f"{parent.root}; refusing to choose one fitted matrix"
                )
            overlaps.append(layer)
    return {layer: union[layer] for layer in sorted(union)}, sorted(set(overlaps))


def merge_lens_artifacts(inputs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Merge layer-sharded lens artifacts and atomically publish the result."""
    if len(inputs) < 2:
        raise ValueError("at least two parent artifacts are required")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    parents = [_load_parent(path) for path in inputs]
    if len({parent.root for parent in parents}) != len(parents):
        raise ValueError("the same parent artifact was supplied more than once")
    _validate_compatibility(parents)
    jacobians, overlap_layers = _union_jacobians(parents)

    import jlens

    reference = parents[0]
    merged_lens = jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=reference.manifest["n_prompts"],
        d_model=reference.manifest["d_model"],
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        lens_path = staging_dir / "lens.pt"
        merged_lens.save(str(lens_path), dtype=reference.serialization_dtype)
        lens_sha256 = sha256_file(lens_path)

        source_corpus = _safe_artifact_file(
            reference.root,
            reference.manifest["corpus_path"],
            field="corpus_path",
        )
        corpus_path = staging_dir / "fit_prompts.jsonl"
        shutil.copyfile(source_corpus, corpus_path)
        if sha256_file(corpus_path) != reference.manifest["corpus_sha256"]:
            raise RuntimeError("copied corpus failed its SHA-256 verification")

        manifest = {
            field: reference.manifest[field] for field in COMPATIBILITY_FIELDS
        }
        manifest.update(
            {
                "artifact_type": "jacobian_lens_layer_union",
                "lens_id": f"jlens-{lens_sha256[:16]}",
                "lens_sha256": lens_sha256,
                "source_layers": sorted(jacobians),
                "corpus_path": corpus_path.name,
                "serialization_dtype": str(reference.serialization_dtype).removeprefix("torch."),
                "created_at": datetime.now(UTC).isoformat(),
                "merge_provenance": {
                    "operation": "exact-layer-union",
                    "overlap_policy": "same dtype, shape, and torch.equal values only",
                    "overlap_layers": overlap_layers,
                    "compatibility_fields": list(COMPATIBILITY_FIELDS),
                    "parents": [
                        {
                            "artifact_name": parent.root.name,
                            "lens_id": parent.manifest["lens_id"],
                            "lens_sha256": parent.lens_sha256,
                            "manifest_sha256": parent.manifest_sha256,
                            "source_layers": parent.manifest["source_layers"],
                        }
                        for parent in parents
                    ],
                },
            }
        )
        write_json_atomic(staging_dir / "manifest.json", manifest)
        os.replace(staging_dir, output_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main() -> None:
    args = _parse_args()
    manifest = merge_lens_artifacts(args.inputs, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

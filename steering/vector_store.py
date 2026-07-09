"""Simple on-disk steering vector registry with metadata bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from schemas.run import SteeringVectorEntry


@dataclass
class SteeringVectorBundle:
    """In-memory representation of a stored steering vector bundle."""

    vector_store_id: str
    trait: str
    method: str
    metadata: dict
    layer_metadata: Dict[int, dict]
    vectors: Dict[int, np.ndarray]

    @property
    def preferred_layers(self) -> List[int]:
        return list(self.metadata.get("preferred_layers") or [])

    @property
    def polarity(self) -> float:
        """Runtime sign calibration for positive-alpha interventions."""

        return coerce_polarity(self.metadata.get("polarity", 1.0))

    def calibrated_vectors(self, polarity: float | None = None) -> Dict[int, np.ndarray]:
        """Return vectors with runtime polarity applied.

        Raw stored vectors keep the extraction equation intact. Evaluation and
        simulation code should use this method so positive alphas reflect the
        calibrated high-trait direction.
        """

        effective_polarity = self.polarity if polarity is None else coerce_polarity(polarity)
        if effective_polarity == 1.0:
            return dict(self.vectors)
        return {
            layer: (vector * effective_polarity).astype(np.float32)
            for layer, vector in self.vectors.items()
        }

    @property
    def vector_hashes(self) -> Dict[int, str]:
        """Content hashes for the loaded, raw vector artifacts."""

        hashes: Dict[int, str] = {}
        for layer, vector in self.vectors.items():
            layer_metadata = self.layer_metadata.get(layer, {})
            recorded = layer_metadata.get("sha256")
            hashes[layer] = str(recorded) if recorded else _array_sha256(vector)
        return hashes


def coerce_polarity(value: object) -> float:
    """Validate runtime vector polarity.

    Polarity is intentionally only a sign flip. Magnitude belongs in alpha or
    per-trait strength settings, not in vector metadata.
    """

    polarity = float(value)
    if polarity not in (-1.0, 1.0):
        raise ValueError(f"Vector polarity must be -1.0 or 1.0, got {value!r}")
    return polarity


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(vector: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(vector)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class VectorStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def save_vectors(
        self,
        trait: str,
        method: str,
        layer_vectors: Dict[int, np.ndarray],
        *,
        model_name: str,
        train_set_hash: str,
        norms: Dict[int, float],
        hyperparameters: Dict[str, str],
        num_train_prompts: int,
        num_eval_prompts: int = 0,
        eval_set_hash: str | None = None,
        vector_store_id: str | None = None,
        layer_diagnostics: Mapping[int, Mapping[str, float]] | None = None,
        polarity: float = 1.0,
    ) -> dict:
        vs_id = vector_store_id or trait
        runtime_polarity = coerce_polarity(polarity)
        diagnostics = layer_diagnostics or {}
        layer_records: List[dict] = []
        for layer, vec in layer_vectors.items():
            vector_path = self.root / f"{vs_id}_layer{layer}.npy"
            np.save(vector_path, vec.astype(np.float32))
            relative_vector_path = vector_path.name
            vector_sha256 = _file_sha256(vector_path)
            entry = SteeringVectorEntry(
                vector_store_id=vs_id,
                trait=trait,
                method=method,
                layer_id=layer,
                vector_path=relative_vector_path,
                train_set_hash=train_set_hash,
                eval_set_hash=eval_set_hash,
                created_at=datetime.utcnow(),
            )
            self._append_entry(entry)
            layer_records.append(
                {
                    "layer_id": layer,
                    "vector_path": relative_vector_path,
                    "sha256": vector_sha256,
                    "norm": float(norms[layer]),
                    "accuracy": None,
                    "accuracy_delta": None,
                    **{
                        key: float(value)
                        for key, value in diagnostics.get(layer, {}).items()
                    },
                }
            )

        metadata = {
            "vector_store_id": vs_id,
            "trait": trait,
            "method": method,
            "model_name": model_name,
            "dataset": {
                "train_size": num_train_prompts,
                "eval_size": num_eval_prompts,
                "train_hash": train_set_hash,
                "eval_hash": eval_set_hash,
            },
            "extraction_hyperparameters": hyperparameters,
            "layers": sorted(layer_records, key=lambda item: item["layer_id"]),
            "preferred_layers": sorted(layer_vectors.keys()),
            "polarity": runtime_polarity,
            "layer_sweep": None,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._write_metadata(trait, metadata)
        return metadata

    def load(
        self, vector_store_id: str, layers: Iterable[int] | None = None
    ) -> SteeringVectorBundle:
        layer_filter = set(layers) if layers else None
        vectors: Dict[int, np.ndarray] = {}
        entries = list(self._iter_entries(vector_store_id))
        if not entries:
            raise ValueError(f"No vectors found for id={vector_store_id}")
        trait = entries[0].trait
        method = entries[0].method
        for entry in entries:
            if layer_filter and entry.layer_id not in layer_filter:
                continue
            vector_path = self.resolve_vector_path(entry.vector_path)
            vector = np.load(vector_path, allow_pickle=False)
            vectors[entry.layer_id] = vector
        if not vectors:
            filter_desc = sorted(layer_filter) if layer_filter else "all"
            raise ValueError(
                f"No vectors found for id={vector_store_id} matching layers={filter_desc}"
            )
        metadata = self._load_metadata(trait)
        if metadata.get("vector_store_id") != vector_store_id:
            raise ValueError(
                f"Metadata for trait {trait} does not match vector_store_id={vector_store_id}"
            )
        layer_metadata = {
            entry["layer_id"]: entry for entry in metadata.get("layers", [])
        }
        for layer_id, vector in vectors.items():
            layer_record = layer_metadata.setdefault(layer_id, {"layer_id": layer_id})
            recorded_hash = layer_record.get("sha256")
            actual_hash = _file_sha256(
                self.resolve_vector_path(
                    layer_record.get("vector_path")
                    or next(
                        entry.vector_path
                        for entry in entries
                        if entry.layer_id == layer_id
                    )
                )
            )
            if recorded_hash and recorded_hash != actual_hash:
                raise ValueError(
                    f"Vector hash mismatch for id={vector_store_id} layer={layer_id}: "
                    f"expected {recorded_hash}, got {actual_hash}"
                )
            layer_record["sha256"] = actual_hash
            layer_record.setdefault("shape", list(vector.shape))
        return SteeringVectorBundle(
            vector_store_id=vector_store_id,
            trait=trait,
            method=method,
            metadata=metadata,
            layer_metadata=layer_metadata,
            vectors=vectors,
        )

    def resolve_vector_path(self, path_value: str | Path) -> Path:
        """Resolve legacy absolute paths and new root-relative vector paths.

        Older metadata captured machine-specific ``/workspace/...`` paths.
        When that path no longer exists, the artifact basename under this
        store is the only portable and unambiguous fallback.
        """

        raw_path = Path(path_value)
        candidates: List[Path] = []
        if raw_path.is_absolute():
            candidates.extend((raw_path, self.root / raw_path.name))
        else:
            candidates.extend((self.root / raw_path, raw_path, self.root / raw_path.name))
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        attempted = ", ".join(str(path) for path in seen)
        raise FileNotFoundError(
            f"Vector artifact {path_value!s} was not found; tried: {attempted}"
        )

    def load_metadata(self, trait: str) -> dict:
        return self._load_metadata(trait)

    def metadata_path(self, trait: str) -> Path:
        return self._metadata_path(trait)

    def write_metadata(self, trait: str, metadata: dict) -> None:
        self._write_metadata(trait, metadata)

    def _append_entry(self, entry: SteeringVectorEntry) -> None:
        with self.index_path.open("a") as fh:
            fh.write(entry.model_dump_json())
            fh.write("\n")

    def _metadata_path(self, trait: str) -> Path:
        return self.root / f"{trait}.meta.json"

    def _load_metadata(self, trait: str) -> dict:
        path = self._metadata_path(trait)
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata for trait {trait} at {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_metadata(self, trait: str, payload: dict) -> None:
        path = self._metadata_path(trait)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def _iter_entries(self, vector_store_id: str):
        if not self.index_path.exists():
            return []
        with self.index_path.open() as fh:
            for line in fh:
                payload = SteeringVectorEntry.model_validate_json(line)
                if payload.vector_store_id == vector_store_id:
                    yield payload

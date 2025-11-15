"""Simple on-disk steering vector registry with metadata bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

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
    ) -> dict:
        vs_id = vector_store_id or trait
        layer_records: List[dict] = []
        for layer, vec in layer_vectors.items():
            vector_path = self.root / f"{vs_id}_layer{layer}.npy"
            np.save(vector_path, vec.astype(np.float32))
            entry = SteeringVectorEntry(
                vector_store_id=vs_id,
                trait=trait,
                method=method,
                layer_id=layer,
                vector_path=str(vector_path),
                train_set_hash=train_set_hash,
                eval_set_hash=eval_set_hash,
                created_at=datetime.utcnow(),
            )
            self._append_entry(entry)
            layer_records.append(
                {
                    "layer_id": layer,
                    "vector_path": str(vector_path),
                    "norm": float(norms[layer]),
                    "accuracy": None,
                    "accuracy_delta": None,
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
            vectors[entry.layer_id] = np.load(entry.vector_path)
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
        return SteeringVectorBundle(
            vector_store_id=vector_store_id,
            trait=trait,
            method=method,
            metadata=metadata,
            layer_metadata=layer_metadata,
            vectors=vectors,
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

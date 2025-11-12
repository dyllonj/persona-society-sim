"""Simple on-disk steering vector registry."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
from uuid import uuid4

import numpy as np

from schemas.run import SteeringVectorEntry


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
        pos_set_hash: str,
        neg_set_hash: str,
        vector_store_id: str | None = None,
    ) -> List[SteeringVectorEntry]:
        vs_id = vector_store_id or f"{trait}-{uuid4().hex[:8]}"
        entries: List[SteeringVectorEntry] = []
        for layer, vec in layer_vectors.items():
            vector_path = self.root / f"{vs_id}_layer{layer}.npy"
            np.save(vector_path, vec)
            entry = SteeringVectorEntry(
                vector_store_id=vs_id,
                trait=trait,
                method=method,
                layer_id=layer,
                vector_path=str(vector_path),
                pos_set_hash=pos_set_hash,
                neg_set_hash=neg_set_hash,
                created_at=datetime.utcnow(),
            )
            entries.append(entry)
            self._append_entry(entry)
        return entries

    def load(self, vector_store_id: str, layers: Iterable[int] | None = None) -> Dict[int, np.ndarray]:
        layer_filter = set(layers) if layers else None
        vectors: Dict[int, np.ndarray] = {}
        for entry in self._iter_entries(vector_store_id):
            if layer_filter and entry.layer_id not in layer_filter:
                continue
            vectors[entry.layer_id] = np.load(entry.vector_path)
        if not vectors:
            raise ValueError(f"No vectors found for id={vector_store_id}")
        return vectors

    def _append_entry(self, entry: SteeringVectorEntry) -> None:
        with self.index_path.open("a") as fh:
            fh.write(entry.model_dump_json())
            fh.write("\n")

    def _iter_entries(self, vector_store_id: str):
        if not self.index_path.exists():
            return []
        with self.index_path.open() as fh:
            for line in fh:
                payload = SteeringVectorEntry.model_validate_json(line)
                if payload.vector_store_id == vector_store_id:
                    yield payload

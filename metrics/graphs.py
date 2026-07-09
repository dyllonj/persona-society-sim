"""Graph utilities for social dynamics analysis."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency for analytics
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover
    nx = None  # type: ignore

from schemas.logs import Edge, GraphSnapshot


def build_graph(edges: Iterable[Edge]) -> nx.DiGraph:
    if nx is None:
        raise ModuleNotFoundError("networkx is required for graph snapshots")
    g = nx.DiGraph()
    for edge in edges:
        if g.has_edge(edge.src, edge.dst):
            data = g[edge.src][edge.dst]
            data["weight"] = float(data.get("weight", 0.0)) + edge.weight
            kind_counts = data.setdefault("kind_counts", {})
            kind_counts[edge.kind] = int(kind_counts.get(edge.kind, 0)) + 1
        else:
            g.add_edge(
                edge.src,
                edge.dst,
                weight=edge.weight,
                kind=edge.kind,
                kind_counts={edge.kind: 1},
                trait_key=edge.trait_key,
                trait_band=edge.trait_band,
                alpha_bucket=edge.alpha_bucket,
            )
    return g


def snapshot_from_edges(
    run_id: str,
    tick: int,
    edges: Iterable[Edge],
    *,
    trait_key: Optional[str] = None,
    band_metadata: Optional[Dict[str, object]] = None,
) -> GraphSnapshot:
    g = build_graph(edges)
    centrality = nx.degree_centrality(g)
    return GraphSnapshot(
        run_id=run_id,
        tick=tick,
        edges=list(edges),
        centrality=centrality,
        trait_key=trait_key,
        band_metadata=band_metadata or {},
    )

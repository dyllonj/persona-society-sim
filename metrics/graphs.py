"""Graph utilities for social dynamics analysis."""

from __future__ import annotations

from typing import Iterable, Tuple

import networkx as nx

from schemas.logs import Edge, GraphSnapshot


def build_graph(edges: Iterable[Edge]) -> nx.Graph:
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge.src, edge.dst, weight=edge.weight, kind=edge.kind)
    return g


def snapshot_from_edges(run_id: str, tick: int, edges: Iterable[Edge]) -> GraphSnapshot:
    g = build_graph(edges)
    centrality = nx.degree_centrality(g)
    return GraphSnapshot(run_id=run_id, tick=tick, edges=list(edges), centrality=centrality)

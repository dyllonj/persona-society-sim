from metrics.graphs import build_graph, snapshot_from_edges
from schemas.logs import Edge


def test_graph_preserves_direction_and_accumulates_repeated_interactions():
    edges = [
        Edge(src="a", dst="b", weight=1.0, kind="message"),
        Edge(src="a", dst="b", weight=2.0, kind="gift"),
        Edge(src="b", dst="a", weight=4.0, kind="message"),
    ]

    graph = build_graph(edges)

    assert graph.is_directed()
    assert graph.number_of_edges() == 2
    assert graph["a"]["b"]["weight"] == 3.0
    assert graph["a"]["b"]["kind_counts"] == {"message": 1, "gift": 1}
    assert graph["b"]["a"]["weight"] == 4.0

    snapshot = snapshot_from_edges("run-1", 0, edges)
    assert len(snapshot.edges) == 3
    assert set(snapshot.centrality) == {"a", "b"}

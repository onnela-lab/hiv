from hiv import km
import networkx as nx


def test_simulate_km() -> None:
    graph = nx.empty_graph(10)
    for _ in range(100):
        graph = km.step(graph, 0.3, 0.7, 0.5)
    assert graph.number_of_nodes() == 10

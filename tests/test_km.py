from hiv import km
import networkx as nx
import pytest


def test_simulate_km() -> None:
    graph, stats = km.simulate(10, 0.3, 0.7, 0.5, 500)
    assert graph.number_of_nodes() == 10
    for value in stats.values():
        assert value.shape == (500,)


def test_wrong_input_graph() -> None:
    with pytest.raises(AssertionError):
        graph = nx.empty_graph(7)
        km.simulate(10, 0.1, 0.1, 0.1, 10, graph)

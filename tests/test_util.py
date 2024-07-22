from hiv import util
import networkx as nx
import pytest


def test_assert_graph_equal() -> None:
    graph1 = nx.erdos_renyi_graph(100, 0.1)
    util.assert_graphs_equal(graph1, graph1)
    graph2 = graph1.copy()
    graph2.remove_node(0)
    with pytest.raises(AssertionError):
        util.assert_graphs_equal(graph1, graph2)


def test_add_nodes_from_update() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5), a="a")
    updated = {2, 3, 7}
    graph.add_nodes_from(updated, a="b")
    for node in updated:
        assert graph.nodes[node]["a"] == "b"
    for node in set(graph) - updated:
        assert graph.nodes[node]["a"] == "a"

    assert graph.number_of_nodes() == 6

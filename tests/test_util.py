from hiv import util
import networkx as nx
import numpy as np
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


def test_pack_unpack_edge() -> None:
    # Pack and unpack, although the order of nodes may change because we sort node
    # labels before packing.
    u, v = np.random.randint(10000, size=(2, 100))
    uv = util.pack_edge(u, v)
    a, b = util.unpack_edge(uv)
    np.testing.assert_array_equal(a, np.minimum(u, v))
    np.testing.assert_array_equal(b, np.maximum(u, v))

    # Re-packing must yield the same result.
    ab = util.pack_edge(a, b)
    np.testing.assert_array_equal(uv, ab)

    # Unpacking again yields the same because the inputs were already sorted.
    i, j = util.unpack_edge(ab)
    np.testing.assert_array_equal(a, i)
    np.testing.assert_array_equal(b, j)


def test_to_from_numpy_graph() -> None:
    nxgraph = nx.barabasi_albert_graph(1000, 4)
    for *_, data in nxgraph.edges(data=True):
        data["type"] = "default"

    npgraph = util.NumpyGraph.from_networkx(nxgraph)
    nxgraph2 = npgraph.to_networkx()
    util.assert_graphs_equal(nxgraph, nxgraph2)

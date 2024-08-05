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
    edges = np.random.randint(10000, size=(100, 2))
    compressed = util.compress_edges(edges)
    edges2 = util.decompress_edges(compressed)
    np.testing.assert_array_equal(edges2, np.sort(edges, axis=-1))

    # Re-packing must yield the same result.
    compressed2 = util.compress_edges(edges2)
    np.testing.assert_array_equal(compressed, compressed2)

    # Unpacking again yields the same because the inputs were already sorted.
    np.testing.assert_allclose(util.decompress_edges(compressed2), edges2)


def test_to_from_numpy_graph() -> None:
    nxgraph = nx.barabasi_albert_graph(1000, 4)
    for *_, data in nxgraph.edges(data=True):
        data["type"] = "default"

    npgraph = util.NumpyGraph.from_networkx(nxgraph)
    nxgraph2 = npgraph.to_networkx()
    util.assert_graphs_equal(nxgraph, nxgraph2)


def test_degree() -> None:
    nxgraph = nx.erdos_renyi_graph(10, 0.1)
    npgraph = util.NumpyGraph.from_networkx(nxgraph)
    np.testing.assert_array_equal(
        npgraph.degrees()["default"],
        [degree for _, degree in sorted(nxgraph.degree)],
    )

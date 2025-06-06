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


@pytest.mark.parametrize("density", [0.1, 0.5])
def test_degree(density: float) -> None:
    nxgraph = nx.erdos_renyi_graph(10, density)
    npgraph = util.NumpyGraph.from_networkx(nxgraph)
    np.testing.assert_array_equal(
        npgraph.degrees()["default"],
        [degree for _, degree in sorted(nxgraph.degree)],
    )
    assert npgraph.nodes.shape == npgraph.degrees(key="default").shape


def test_flatten_dict() -> None:
    X1 = {
        "a": np.random.normal(size=(10,)),
        "b": np.random.normal(size=(10, 5)),
        "c": np.random.normal(size=(10, 3, 7)),
    }
    estimator = util.FlattenDict().fit(X1)
    y1 = estimator.transform(X1)
    assert y1.shape == (10, 1 + 5 + 3 * 7)

    X2 = estimator.inverse_transform(y1)
    assert tuple(X1) == tuple(X2)
    for key, value in X1.items():
        np.testing.assert_allclose(value, X2[key])

    X3 = dict(reversed(X1.items()))
    y3 = estimator.transform(X3)
    np.testing.assert_allclose(y1, y3)


@pytest.mark.parametrize(
    "proba, factor, expected",
    [
        (1e-6, 7, 7e-6),
        (0.01, 50, 0.394994),
    ],
)
def test_transform_proba_discrete(proba: float, factor: float, expected: float) -> None:
    actual = util.transform_proba_discrete(proba, factor)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-6)
    rec = util.transform_proba_discrete(actual, 1 / factor)
    np.testing.assert_allclose(proba, rec)

    # Simulate the process and do a z-score check if there is any variation (not all 0
    # or 1).
    events = (np.random.uniform(size=(1000, factor)) < proba).any(axis=-1)
    mean = events.mean()
    std = events.std() / (events.size - 1) ** 0.5
    if std > 0:
        z = (mean - actual) / std
        assert np.abs(z) < 2.5


@pytest.mark.parametrize(
    "rate, factor, expected",
    [
        (1, 3, 0.950213),
    ],
)
def test_transform_proba_continuous(
    rate: float, factor: float, expected: float
) -> None:
    actual = util.transform_proba_continuous(rate, factor)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    events = np.random.exponential(rate * factor, 1000) > 0
    mean = events.mean()
    std = events.std() / (events.size - 1) ** 0.5
    if std > 0:
        z = (mean - actual) / std
        assert np.abs(z) < 2

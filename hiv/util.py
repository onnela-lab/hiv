import contextlib
import networkx as nx
import numpy as np
import typing
from time import time


class Timer:
    """
    Timer as a context manager with different keys.
    """

    def __init__(self):
        self.times = {}

    @contextlib.contextmanager
    def __call__(self, key):
        start = time()
        yield
        self.times[key] = time() - start


def assert_graphs_equal(actual: nx.Graph, expected: nx.Graph) -> None:
    """
    Assert that two graphs have the same nodes, edges, and attributes.
    """
    assert dict(actual.nodes(data=True)) == dict(expected.nodes(data=True))
    assert {tuple(edge): data for *edge, data in actual.edges(data=True)} == {
        tuple(edge): data for *edge, data in expected.edges(data=True)
    }


def to_np_dict(
    x: dict[typing.Hashable, typing.Iterable]
) -> dict[typing.Hashable, np.ndarray]:
    return {key: np.asarray(value) for key, value in x.items()}


def pack_edge(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Pack node indices `u` and `v` into a single edge identifier obtained by
    concatenating the numbers.
    """
    uvmax = 0xFFFFFFFF
    assert u.max() <= uvmax and v.max() <= uvmax
    u, v = np.minimum(u, v).astype(np.uint64), np.maximum(u, v).astype(np.uint64)
    uv = (u << 32) + v
    assert uv.dtype == np.uint64
    return uv


def unpack_edge(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack a compressed edge.
    """
    assert uv.dtype == np.uint64
    mask = 0xFFFFFFFF00000000
    u = (uv & mask) >> 32
    v = uv & ~mask
    return u, v


class NumpyGraph:
    """
    Graph represented as numpy arrays.

    Args:
        nodes: Set of nodes.
        edges: Mapping from edge types to sets of edges.
    """

    def __init__(
        self, nodes: np.ndarray = None, edges: dict[str, np.ndarray] = None
    ) -> None:
        self.nodes = nodes
        self.edges = edges

    def to_networkx(self) -> nx.Graph:
        """
        Convert the graph to a networkx graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        for key, edges in self.edges.items():
            graph.add_edges_from(edges, type=key)
        return graph

    @classmethod
    def from_networkx(cls, graph: nx.Graph):
        """
        Create a graph from a networkx graph.
        """
        nodes = np.asarray(list(graph.nodes))
        assert np.issubdtype(nodes.dtype, int)
        edges = {}
        for *edge, data in graph.edges(data=True):
            edges.setdefault(data["type"], []).append(edge)
        edges = {key: np.asarray(value) for key, value in edges.items()}
        return cls(nodes, edges)

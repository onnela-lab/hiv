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


def compress_edges(uv: np.ndarray) -> np.ndarray:
    """
    Pack an edge list with shape `(num_edges, 2)` to a compressed edge with shape
    `(num_edges,)`.
    """
    assert uv.size == 0 or uv.max() < 0xFFFFFFFF
    uv.sort(axis=-1)
    u, v = uv.T
    uv = (u.astype(np.uint64) << 32) + v.astype(np.uint64)
    assert uv.dtype == np.uint64
    return uv


def decompress_edges(uv: np.ndarray) -> np.ndarray:
    """
    Unpack a compressed edge with shape `(num_edges,)` to an edge list with shape
    `(num_edges, 2)`.
    """
    assert uv.dtype == np.uint64
    return np.transpose([(uv & 0xFFFFFFFF00000000) >> 32, uv & 0x00000000FFFFFFFF])


def candidates_to_edges(candidates: np.ndarray) -> np.ndarray:
    candidates = np.random.permutation(candidates)
    if candidates.size % 2:
        candidates = candidates[1:]
    edges = candidates.reshape((-1, 2))
    return compress_edges(edges)


class NumpyGraph:
    """
    Graph represented by numpy arrays. This implementation is fast for batch updates of
    nodes or edges and slow for iterative updates.

    Args:
        nodes: Set of nodes.
        compressed: Mapping from edge types to compressed edge sets.
    """

    def __init__(
        self, nodes: np.ndarray = None, edges: dict[str, np.ndarray] = None
    ) -> None:
        self.nodes = nodes
        self.edges = edges or {}

    def copy(self) -> "NumpyGraph":
        """
        Shallow copy of the graph.
        """
        return self.__class__(self.nodes, self.edges.copy())

    def degrees(self, key=None) -> np.ndarray:
        if key is None:
            return {key: self.degrees(key) for key in self.edges}
        edges = decompress_edges(self.edges[key])
        connected_nodes, connected_degrees = np.unique(edges, return_counts=True)
        degrees = np.zeros_like(self.nodes)
        degrees[np.searchsorted(self.nodes, connected_nodes)] = connected_degrees
        return degrees

    def to_networkx(self) -> nx.Graph:
        """
        Convert the graph to a networkx graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        for key, edges in self.edges.items():
            graph.add_edges_from(decompress_edges(edges), type=key)
        return graph

    def validate(self) -> nx.Graph:
        # Check nodes are sorted.
        np.testing.assert_array_less(
            0, np.diff(self.nodes), err_msg="Node labels must be sorted."
        )

        # Check there are no edges that do not have corresponding nodes.
        for key, compressed in self.edges.items():
            decompressed = decompress_edges(compressed)
            has_nodes = np.isin(decompressed, self.nodes).all(axis=-1)
            assert has_nodes.all(), f"Edges with type {key} have missing nodes."

        # Check that edges are unique.
        concatenated = np.concatenate(list(self.edges.values()))
        nunique = np.unique(concatenated).size
        assert concatenated.size == nunique, "Edges are not unique."

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> "NumpyGraph":
        """
        Create a graph from a networkx graph.
        """
        nodes = np.asarray(list(graph.nodes))
        assert np.issubdtype(nodes.dtype, int)
        edges = {}
        for *edge, data in graph.edges(data=True):
            edges.setdefault(data.get("type", "default"), []).append(edge)
        edges = {key: compress_edges(np.asarray(value)) for key, value in edges.items()}
        return cls(nodes, edges)

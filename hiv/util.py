import contextlib
import networkx as nx  # type: ignore
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
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
    """
    Randomly pair candidates to create edges. If the number of candidates is odd, one of
    them is dropped at random.

    Args:
        candidates: Node identifiers to pair with shape `(n,)`.

    Returns:
        Compressed edge list with shape `(n // 2,)`.
    """
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
        self,
        nodes: np.ndarray | None = None,
        edges: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.nodes = np.empty((), dtype=int) if nodes is None else nodes
        self.edges = edges or {}

    def copy(self) -> "NumpyGraph":
        """
        Shallow copy of the graph.
        """
        return self.__class__(self.nodes, self.edges.copy())

    @typing.overload
    def degrees(self, key: str) -> np.ndarray: ...

    @typing.overload
    def degrees(self, key: None) -> dict[str, np.ndarray]: ...

    def degrees(self, key: str | None = None) -> np.ndarray | dict[str, np.ndarray]:
        """
        Evaluate the degree of nodes.

        Args:
            key: Edge key to evaluate the degree for.

        Returns:
            If `key` is given, vector of degrees for keyed edges corresponding to
            :attr:`nodes`. If `key` is not given, a dictionary mapping edge keys to the
            corresponding degree vector.
        """
        if key is None:
            return {key: self.degrees(key) for key in self.edges}
        edges = decompress_edges(self.edges[key])
        connected_nodes, connected_degrees = np.unique(edges, return_counts=True)
        assert connected_nodes.size <= self.nodes.size
        degrees = np.zeros_like(self.nodes)
        idx = np.isin(self.nodes, connected_nodes)
        degrees[idx] = connected_degrees
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

    def validate(self) -> None:
        """
        Validate the structure of the graph, ensuring that

        - node labels are sorted,
        - edges do not refer to missing nodes,
        - and edges are unique across keys (i.e., edges can only exist in one layer of a
          multi-layer graph).
        """
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


def _validate_shapes(
    X: dict[str, np.ndarray], expected_shapes: dict[str, tuple] | None = None
) -> tuple[int, dict[str, tuple]]:
    if expected_shapes:
        assert set(X) == set(expected_shapes)

    n_samples = None
    actual_shapes = {}
    for key, value in X.items():

        # Validate consistent number of samples.
        n, *actual_shape = value.shape
        if n_samples is None:
            n_samples = n
        elif n_samples != n:
            raise ValueError  # pragma: no cover

        # Validate shapes.
        actual_shape = tuple(actual_shape)
        if expected_shapes:
            assert expected_shapes[key] == actual_shape
        else:
            actual_shapes[key] = actual_shape

    return n_samples, actual_shapes


class FlattenDict(TransformerMixin, BaseEstimator):
    """
    Flatten a dictionary of arrays to a matrix.
    """

    def fit(self, X: dict[str, np.ndarray], y: None = None) -> "FlattenDict":
        _, self.shapes_ = _validate_shapes(X)
        return self

    def transform(self, X: dict[str, np.ndarray]) -> np.ndarray:
        if not hasattr(self, "shapes_"):
            raise NotFittedError  # pragma: no cover
        n_samples, _ = _validate_shapes(X, self.shapes_)
        parts = [X[key].reshape((n_samples, -1)) for key in self.shapes_]
        return np.concatenate(parts, -1)

    def inverse_transform(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not hasattr(self, "shapes_"):
            raise NotFittedError  # pragma: no cover

        n_samples, actual_size = X.shape
        assert actual_size == sum([np.prod(shape) for shape in self.shapes_.values()])

        offset = 0
        result = {}
        for key, shape in self.shapes_.items():
            size = int(np.prod(shape))
            result[key] = X[:, offset : offset + size].reshape((n_samples, *shape))
            offset += size
        return result


def transform_proba_discrete(proba: float, factor: float) -> float:
    """
    Transform the probability for an event to happen on one timescale to another,
    assuming discrete-time dynamics.

    Args:
        proba: Original probability for the event to happen.
        factor: Ratio of timescales with transformation to a longer scale indicated by
            :code:`factor > 1`, e.g., :code:`factor = 7` for day-to-week transformation.

    Returns:
        Transformed probability.
    """
    # The probability for the event to happen at least once is 1 - probability that the
    # event does not happen at all, i.e., at_least_once = 1 - (1 - proba) ** factor. We
    # use log1p formulation for small probabilities/large factor changes.
    not_proba = np.exp(factor * np.log1p(-proba))
    return np.exp(np.log1p(-not_proba))


def transform_proba_continuous(rate: float, factor: float) -> float:
    """
    Transform the probability for an event to happen on one timescale in continuous time
    to the event happening in a discrete time window.

    Args:
        rate: Rate at which events happen.
        factor: Length of the time window.

    Returns:
        Probability that an event happens in the window.
    """
    rate = np.asarray(rate)
    factor = np.asarray(factor)
    return 1 - np.exp(-rate * factor)

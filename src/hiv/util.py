import collectiontools
import contextlib
import networkx as nx  # type: ignore
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from typing import Any, Callable, Hashable, Iterable
from time import time


DEBUG = "CI" in os.environ


class Timer:
    """
    Timer as a context manager with different keys.
    """

    def __init__(self):
        self.times: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, key: str):
        start = time()
        yield
        self.times[key] = time() - start


def to_np_dict(
    x: dict[Hashable, Iterable],
    cond: Callable[[Hashable], bool] | None = None,
) -> dict[Hashable, np.ndarray]:
    cond = cond or (lambda _: True)
    result = {}
    for key, value in x.items():
        if cond(key):
            try:
                value = np.asarray(value)
            except Exception as ex:  # pragma: no cover
                raise ValueError(f"Failed to convert '{key}' to numpy array.") from ex
        result[key] = value
    return result


def compress_edges(uv: np.ndarray) -> np.ndarray:
    """
    Pack an edge list with shape `(num_edges, 2)` to a compressed edge with shape
    `(num_edges,)`.
    """
    assert uv.ndim == 2
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
    assert uv.dtype == np.uint64, f"Expected dtype {np.uint64}, got {uv.dtype}."
    return np.transpose(
        [(uv & 0xFFFFFFFF00000000) >> 32, uv & 0x00000000FFFFFFFF]
    ).astype(np.uint32)


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


def coerce_matching_shape(
    x: np.ndarray, attributes: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Validate the shape of all attributes matches the shape of the input. If an
    attribute is a scalar, it is broadcast to the shape of the input.

    Args:
        x: Input array.
        attributes: Dictionary of attributes whose values must have shape matching the
            input.

    Returns:
        Attributes with coerced shapes.
    """
    assert x.ndim == 1, "Input must be a vector."

    coerced = {}
    for key, value in attributes.items():
        if np.size(value) == 1:
            # We make a copy because the result is otherwise not writeable.
            value = np.broadcast_to(value, x.shape).copy()
        assert x.shape == value.shape, (
            f"Attribute '{key}' with shape '{value.shape}' does not match input with "
            f"shape '{x.shape}'."
        )
        coerced[key] = value
    return coerced


def as_immutable_view(x: np.ndarray) -> np.ndarray:
    y = x.view()
    y.flags["WRITEABLE"] = False
    return y


class NumpyGraph:
    """
    Graph represented by numpy arrays. This implementation is fast for batch updates of
    nodes or edges and slow for iterative updates.

    Args:
        nodes: Set of nodes.
        compressed: Set of compressed edges.
        node_attrs: Mapping from names to node attributes.
        edge_attrs: Mapping from names to edge attributes.
        attrs: Mapping from names to graph attributes.
    """

    def __init__(
        self,
        *,
        nodes: np.ndarray | None = None,
        edges: np.ndarray | None = None,
        node_attrs: dict[str, np.ndarray | np.dtype] | None = None,
        edge_attrs: dict[str, np.ndarray | np.dtype] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._nodes = np.empty((0,), dtype=np.uint32) if nodes is None else nodes
        self._edges = np.empty((0,), dtype=np.uint64) if edges is None else edges
        self.node_attrs = self._as_attribute_dict(node_attrs)
        self.edge_attrs = self._as_attribute_dict(edge_attrs)
        self.attrs = attrs or {}

        # Run very basic checks that are cheap in the constructor.
        self.validate_shapes()

    @property
    def nodes(self):
        return as_immutable_view(self._nodes)

    @property
    def edges(self):
        return as_immutable_view(self._edges)

    def _as_attribute_dict(
        self, value: dict[str, np.ndarray] | np.dtype | None
    ) -> dict[str, np.ndarray | None]:
        if value is None:
            return {}  # pragma: no cover
        assert isinstance(value, dict)
        return {
            key: (
                dtype_or_array
                if isinstance(dtype_or_array, np.ndarray)
                else np.empty((0,), dtype=dtype_or_array)
            )
            for key, dtype_or_array in value.items()
        }

    def copy(self) -> "NumpyGraph":
        """
        Copy of the graph. This is a shallow copy in the sense that numpy arrays are
        *not* copied, but any container dictionaries *are* copied.
        """
        return self.__class__(
            nodes=self.nodes,
            edges=self.edges,
            node_attrs=self.node_attrs.copy(),
            edge_attrs=self.edge_attrs.copy(),
            attrs=self.attrs.copy(),
        )

    def subgraph(self, nodes: np.ndarray) -> "NumpyGraph":  # pragma: no cover
        """
        Create a copy of the graph that includes only a subset of nodes.

        Args:
            nodes: Nodes to include in the graph.

        Returns:
            Copy of the graph that includes only a subset of nodes.
        """
        missing = np.setdiff1d(nodes, self._nodes)
        assert (
            missing.size == 0
        ), f"{missing.size} nodes in the subset are missing from the graph."
        node_mask = np.isin(self._nodes, nodes)
        edge_mask = np.isin(decompress_edges(self._edges), nodes).any(axis=-1)
        return self.__class__(
            nodes=nodes,
            edges=self._edges[edge_mask],
            node_attrs={
                key: value[node_mask] for key, value in self.node_attrs.items()
            },
            edge_attrs={
                key: value[edge_mask] for key, value in self.edge_attrs.items()
            },
            attrs=self.attrs.copy(),
        )

    def _add(
        self,
        x_new: np.ndarray,
        attrs_new: dict[str, np.ndarray],
        x_old: np.ndarray,
        attrs_old: dict[str, np.ndarray | None],
    ) -> None:
        attrs_new = coerce_matching_shape(x_new, attrs_new)
        assert (
            not DEBUG or np.intersect1d(x_new, x_old).size == 0
        ), "Duplicate values detected."

        keys_old = set(attrs_old)
        keys_new = set(attrs_new)
        assert keys_old == keys_new, (
            f"Existing attributes '{keys_old}' do not match new attributes "
            f"'{keys_new}'."
        )

        x_result = np.concatenate([x_old, x_new])
        attrs_result = {}
        for key, value in attrs_new.items():
            if attrs_old[key].size:
                value = np.concatenate([attrs_old[key], value])
            attrs_result[key] = value

        return x_result, attrs_result

    def add_nodes(self, nodes: np.ndarray, **kwargs) -> None:
        """Add edges and their attributes.

        Args:
            edges: Compressed edges.
            kwargs: Edge attributes.
        """
        self._nodes, self.node_attrs = self._add(
            nodes, kwargs, self._nodes, self.node_attrs
        )

    def _filter(
        self, x: np.ndarray, attrs: dict[str, np.ndarray], fltr: np.ndarray
    ) -> None:
        return x[fltr], {
            key: value if value is None else value[fltr] for key, value in attrs.items()
        }

    def filter_nodes(self, fltr: np.ndarray) -> None:
        self._nodes, self.node_attrs = self._filter(self._nodes, self.node_attrs, fltr)
        # We also need to filter out any edges that no longer have one of its vertices.
        edges = decompress_edges(self._edges)
        both_exist = np.isin(edges, self._nodes).all(axis=-1)
        self.filter_edges(both_exist)

    def add_edges(self, edges: np.ndarray, **kwargs) -> None:
        """Add edges and their attributes.

        Args:
            edges: Compressed edges.
            kwargs: Edge attributes.
        """
        self._edges, self.edge_attrs = self._add(
            edges, kwargs, self._edges, self.edge_attrs
        )

    def filter_edges(self, fltr: np.ndarray) -> None:
        self._edges, self.edge_attrs = self._filter(self._edges, self.edge_attrs, fltr)

    def degrees(
        self, *, key: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None
    ) -> np.ndarray:
        """
        Evaluate the degree of nodes.

        Args:
            key: Predicate to evaluate if an edge should be included in the evaluation
                based on its attributes.

        Returns:
            Vector of degrees for keyed edges corresponding to :attr:`nodes`.
        """
        # Restrict edge set based on predicate and decompress edges.
        edges = self.edges
        if key:
            edges = edges[key(self.edge_attrs)]
        edges = decompress_edges(edges)

        # Get indices of non-isolated nodes and the corresponding number of connections.
        connected_nodes, connected_degrees = np.unique(edges, return_counts=True)
        assert connected_nodes.size <= self.nodes.size

        # Get a mask of nodes that are non-isolated, and set the counts.
        degrees = np.zeros_like(self.nodes)
        idx = np.isin(self.nodes, connected_nodes)
        degrees[idx] = connected_degrees
        return degrees

    def to_networkx(self) -> nx.Graph:
        """
        Convert the graph to a networkx graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(
            (node, {key: value[i] for key, value in self.node_attrs.items()})
            for i, node in enumerate(self.nodes)
        )
        graph.add_edges_from(
            (
                *decompress_edges(edge),
                {key: value[i] for key, value in self.edge_attrs.items()},
            )
            for i, edge in enumerate(self.edges)
        )
        return graph

    def validate(self) -> None:
        """
        Validate the structure of the graph, ensuring that

        - node labels are sorted
        - edges do not refer to missing nodes
        - edges are unique across keys (i.e., edges can only exist in one layer of a
          multi-layer graph)
        - node and edge attributes match the structures of nodes and edges, respectively
        """
        self.validate_shapes()

        # Check nodes are sorted.
        np.testing.assert_array_less(
            0, np.diff(self.nodes), err_msg="Node labels must be sorted."
        )

        # Check there are no edges that do not have corresponding nodes.
        decompressed = decompress_edges(self.edges)
        has_nodes = np.isin(decompressed, self.nodes).all(axis=-1)
        assert has_nodes.all(), f"Edges have missing nodes."
        assert np.unique(self.edges).size == self.edges.size

    def validate_shapes(self) -> None:
        """Validate that arrays have correct shapes."""
        assert (
            self.edges.ndim == 1
        ), f"Compressed edges must be a vector, got shape {self.edges.shape}."
        assert (
            self.nodes.ndim == 1
        ), f"Nodes must be a vector, got shape {self.nodes.shape}."

        coerce_matching_shape(self.nodes, self.node_attrs)
        coerce_matching_shape(self.edges, self.edge_attrs)

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> "NumpyGraph":
        """
        Create a graph from a networkx graph.
        """
        data: dict

        # Construct nodes and node attributes.
        nodes = []
        node_attrs: dict[str, list] = {}
        for node, data in graph.nodes(data=True):
            nodes.append(node)
            for key, value in data.items():
                node_attrs.setdefault(key, []).append(value)

        # Construct edges and edge attributes.
        edges = []
        edge_attrs: dict[str, list] = {}
        for *edge, data in graph.edges(data=True):
            edges.append(edge)
            for key, value in data.items():
                edge_attrs.setdefault(key, []).append(value)

        return cls(
            nodes=np.asarray(nodes),
            edges=compress_edges(np.asarray(edges)),
            node_attrs=collectiontools.map_values(np.asarray, node_attrs),
            edge_attrs=collectiontools.map_values(np.asarray, edge_attrs),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{super().__repr__()} with {self.nodes.size} nodes and "
            f"{({key: edges.size for key, edges in self.edges.items()})} edges"
        )


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

    assert n_samples is not None, "Number of samples could not be determined."
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
    if proba == 0:  # pragma: no cover
        return 0
    # The probability for the event to happen at least once is 1 - probability that the
    # event does not happen at all, i.e., at_least_once = 1 - (1 - proba) ** factor. We
    # use log1p formulation for small probabilities/large factor changes.
    not_proba = np.exp(factor * np.log1p(-proba))
    return np.exp(np.log1p(-not_proba))


def transform_proba_continuous(
    rate: float | np.ndarray, factor: float | np.ndarray
) -> float | np.ndarray:
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


def assert_graphs_equal(
    actual: nx.Graph | NumpyGraph, expected: nx.Graph | NumpyGraph
) -> None:
    """
    Assert that two graphs have the same nodes, edges, and attributes.
    """
    if actual is expected:
        return
    if isinstance(actual, NumpyGraph):
        actual = actual.to_networkx()
    if isinstance(expected, NumpyGraph):
        expected = expected.to_networkx()
    assert dict(actual.nodes(data=True)) == dict(expected.nodes(data=True))
    assert {tuple(edge): data for *edge, data in actual.edges(data=True)} == {
        tuple(edge): data for *edge, data in expected.edges(data=True)
    }

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

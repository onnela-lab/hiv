import networkx as nx
import numpy as np


class Constraint:
    def is_valid(self, value) -> bool:
        raise NotImplementedError


class Interval(Constraint):
    def __init__(self, low, high, type=None) -> None:
        assert low is None or high is None or low < high
        self.low = low
        self.high = high
        self.type = type

    def is_valid(self, value) -> bool:
        return (
            (self.low is None or self.low <= value)
            and (self.high is None or value <= self.high)
            and (self.type is None or isinstance(value, self.type))
        )

    def __repr__(self) -> str:
        args = [
            f"low={self.low}",
            f"high={self.high}",
        ]
        if self.type is not None:
            args.append(f"type={self.type}")
        return f"{self.__class__.__name__}({', '.join(args)})"


class UnitInterval(Interval):
    def __init__(self) -> None:
        super().__init__(0, 1)


class Simulator:
    arg_constraints: dict[str, Constraint] = {}

    def init(self) -> nx.Graph:
        raise NotImplementedError

    def step(self, graph: nx.Graph) -> nx.Graph:
        raise NotImplementedError

    def run(self, graph: nx.Graph, num_steps: int) -> nx.Graph:
        for _ in range(num_steps):
            graph = self.step(graph)
        return graph

    def validate_args(self) -> None:
        for arg, constraint in self.arg_constraints.items():
            if not constraint.is_valid(getattr(self, arg)):
                raise ValueError(
                    f"Argument `{arg}` does not satisfy constraint `{constraint}`."
                )

    def evaluate_summaries(
        self, graph0: nx.Graph, graph1: nx.Graph
    ) -> dict[str, float]:
        """
        Evaluate summary statistics.

        Args:
            graph0: First graph observation.
            graph1: Second graph observation.

        Returns:
            Mapping of summary statistics.
        """
        raise NotImplementedError


def add_edges_from_candidates(
    graph: nx.Graph, candidates, shuffle: bool = True, **kwargs
) -> np.ndarray:
    """
    Add edges from an iterable of candidates to be paired up akin to a configuration
    model. Edges are only added if they do not already exist.

    Args:
        graph: Graph to add edges to.
        candidates: Array-like candidates to pair up.
        shuffle: Shuffle candidates before pairing.
        **kwargs: Attributes for each edge.

    Returns:
        Edges added to the graph as an edge list with shape `(num_candidates // 2, 2)`.
    """
    if shuffle:
        candidates = np.random.permutation(candidates)
    if len(candidates) % 2:
        candidates = candidates[1:]
    edges = np.reshape(candidates, (-1, 2))
    graph.add_edges_from((*edge, kwargs) for edge in edges if not graph.has_edge(*edge))
    return edges


def number_of_nodes(graph, predicate=None, **kwargs) -> int:
    """
    Evaluate the number of nodes satisfying the predicate or matching the keyword
    arguments.
    """
    if predicate is None:
        predicate = lambda data: all(  # noqa: E731
            data[key] == value for key, value in kwargs.items()
        )
    return sum(1 for _, data in graph.nodes(data=True) if predicate(data))

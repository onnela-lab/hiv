import networkx as nx
import numpy as np
from .util import candidates_to_edges, NumpyGraph, Timer, decompress_edges


def number_of_nodes(graph: nx.Graph, predicate=None, **kwargs) -> int:
    """
    Evaluate the number of nodes satisfying the predicate or matching the keyword
    arguments.
    """
    if predicate is None:
        predicate = lambda data: all(  # noqa: E731
            data[key] == value for key, value in kwargs.items()
        )
    return sum(1 for _, data in graph.nodes(data=True) if predicate(data))


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


class UniversalSimulator:
    r"""
    Universal discrete-time simulator for sexual contact networks, reducing to different
    models depending on parameterization.

    Args:
        n: Expected number of nodes.
        mu: Probability for a node to emigrate and immigration rate `n * mu`.
        sigma: Probability for a steady relationship to dissolve.
        rho: Propensity for a steady relationship to form.
        xi: Propensity for monogamy.
        omega0: Probability for a casual relationship to form between singles.
        omega1: Probability for a casual relationship to form between non-singles.
    """

    arg_constraints = {
        "n": Interval(0, None),
        "mu": UnitInterval(),
        "sigma": UnitInterval(),
        "rho": UnitInterval(),
        "omega0": UnitInterval(),
        "omega1": UnitInterval(),
        "xi": UnitInterval(),
    }

    def __init__(
        self,
        *,
        n: float,
        mu: float,
        sigma: float,
        rho: float,
        xi: float,
        omega0: float,
        omega1: float,
    ) -> None:
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.omega0 = omega0
        self.omega1 = omega1
        self.xi = xi
        self.validate_args()

    def run(self, graph: NumpyGraph, num_steps: int) -> NumpyGraph:
        for _ in range(num_steps):
            graph = self.step(graph)
        return graph

    def validate_args(self) -> None:
        for arg, constraint in self.arg_constraints.items():
            if not constraint.is_valid(getattr(self, arg)):
                raise ValueError(
                    f"Argument `{arg}` does not satisfy constraint `{constraint}`."
                )

    def init(self) -> NumpyGraph:
        return NumpyGraph(np.arange(round(self.n)))

    def step(self, graph: NumpyGraph, return_times: bool = False) -> NumpyGraph:
        label_offset = (graph.nodes.max() + 1) if graph.nodes.size else 0
        timer = Timer()

        # Remove nodes with probability mu.
        with timer("remove_nodes"):
            num_keep = np.random.binomial(graph.nodes.size, 1 - self.mu)
            graph.nodes = np.random.choice(graph.nodes, num_keep, replace=False)

        # Remove edges incident on a removed node.
        if "steady" in graph.edges:
            with timer("remove_lingering_edges"):
                compressed_steady = graph.edges["steady"]
                steady = decompress_edges(compressed_steady)
                # We can't use assume_unique here because node indices may appear
                # repeatedly in the edge list.
                graph.edges["steady"] = compressed_steady[
                    np.isin(steady, graph.nodes).any(axis=-1)
                ]

        # Add new nodes that have migrated in.
        with timer("add_nodes"):
            num_new_nodes = np.random.poisson(self.n * self.mu)
            graph.nodes = np.concatenate(
                [graph.nodes, label_offset + np.arange(num_new_nodes)]
            )

        # If there are no nodes, there's nothing else to be done.
        if not graph.nodes.size:
            return graph

        # Remove steady relationships with probability sigma.
        if "steady" in graph.edges:
            with timer("remove_steady_edges"):
                num_edges = graph.edges["steady"].shape[0]
                num_keep = np.random.binomial(num_edges, 1 - self.sigma)
                graph.edges["steady"] = np.random.choice(
                    graph.edges["steady"], size=num_keep, replace=False
                )

        with timer("add_steady_edges"):
            # Seek steady edges with probability depending on being single.
            if "steady" in graph.edges:
                # We can't use assume_unique here because node indices may appear
                # repeatedly in the edge list.
                is_partnered = np.isin(
                    graph.nodes, decompress_edges(graph.edges["steady"])
                )
                proba = np.where(is_partnered, self.rho * (1 - self.xi), self.rho)
            else:
                proba = self.rho
            candidates = graph.nodes[np.random.uniform(size=graph.nodes.size) < proba]

            # Add new edges and deduplicate if already in a relationship.
            new_steady_edges = candidates_to_edges(candidates)
            if "steady" in graph.edges:
                graph.edges["steady"] = np.union1d(
                    graph.edges["steady"], new_steady_edges
                )
            else:
                graph.edges["steady"] = new_steady_edges

        # Add casual relationships. Because we have already removed all casual
        # relationships from the previous iteration, any edge is a steady relationship,
        # and the vertices of any edge are partnered up.
        with timer("add_casual_edges"):
            # Seek casual edges with probability depending on being single. We don't
            # need to check if there are steady edges because we just created them.
            is_partnered = np.isin(graph.nodes, decompress_edges(graph.edges["steady"]))
            proba = np.where(is_partnered, self.omega1, self.omega0)
            candidates = graph.nodes[np.random.uniform(size=graph.nodes.size) < proba]

            # Add new edges and exclude edges that already exist in the steady edge set.
            # We need to first compress the edges to allow the set operations.
            casual_edges = candidates_to_edges(candidates)
            graph.edges["casual"] = np.setdiff1d(
                casual_edges, graph.edges["steady"], assume_unique=True
            )

        return (graph, timer.times) if return_times else graph

    def evaluate_summaries(
        self, graph0: NumpyGraph, graph1: NumpyGraph
    ) -> dict[str, float]:
        steady_edges0 = graph0.edges["steady"]
        steady_edges1 = graph1.edges["steady"]

        # Evaluate number of nodes and nodes with casual relationships by relationship
        # status.
        num_nodes = {"single": 0, "partnered": 0}
        num_nodes_with_casual = {"single": 0, "partnered": 0}
        for graph in [graph0, graph1]:
            is_partnered = np.in1d(graph.nodes, decompress_edges(graph.edges["steady"]))
            num_partnered = is_partnered.sum()
            num_nodes["partnered"] += num_partnered
            num_nodes["single"] += graph.nodes.size - num_partnered.sum()

            has_casual = np.in1d(graph.nodes, decompress_edges(graph.edges["casual"]))
            num_nodes_with_casual["partnered"] = is_partnered @ has_casual
            num_nodes_with_casual["single"] = (~is_partnered) @ has_casual

        return {
            "frac_retained_nodes": np.intersect1d(
                graph0.nodes, graph1.nodes, assume_unique=True
            ).size
            / graph0.nodes.size,
            "frac_retained_steady_edges": np.intersect1d(
                steady_edges0, steady_edges1, assume_unique=True
            ).size
            / max(len(steady_edges0), 1),  # noqa: E131
            "frac_single_with_casual": num_nodes_with_casual["single"]
            / max(num_nodes["single"], 1),
            "frac_paired_with_casual": num_nodes_with_casual["partnered"]
            / max(num_nodes["partnered"], 1),
            "frac_paired": num_nodes["partnered"]
            / (graph0.nodes.size + graph1.nodes.size),
            "num_steady_edges": len(steady_edges1),
            "num_casual_edges": graph1.edges["casual"].size,
            "num_nodes": graph1.nodes.size,
        }

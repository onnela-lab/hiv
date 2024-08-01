import networkx as nx
import numpy as np
from .util import Timer


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


def degree(graph: nx.Graph):
    for node, neighbors in graph._adj.items():
        yield node, len(neighbors)


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

    def init(self) -> nx.Graph:
        return nx.empty_graph(round(self.n))

    def step(self, graph: nx.Graph, return_times: bool = False) -> nx.Graph:
        label_offset = max(graph, default=-1) + 1
        timer = Timer()

        # Remove nodes with probability mu.
        with timer("remove_nodes"):
            nodes = np.asarray(list(graph))
            nodes_to_remove = nodes[
                np.random.binomial(1, self.mu, nodes.shape).astype(bool)
            ]
            graph.remove_nodes_from(nodes_to_remove)

        # Add new nodes that have migrated in.
        with timer("add_nodes"):
            num_new_nodes = np.random.poisson(self.n * self.mu)
            graph.add_nodes_from(label_offset + np.arange(num_new_nodes))

        # If there are no nodes, there's nothing else to be done.
        if not graph.number_of_nodes():
            return graph

        # Remove steady relationships with probability sigma and all casual
        # relationships.
        with timer("edges_to_array"):
            edges = np.asarray(
                [
                    (*edge, data["type"] == "casual")
                    for *edge, data in graph.edges(data=True)
                ]
            )
        if edges.size:
            with timer("remove_edges"):
                edges_to_remove = edges[
                    np.random.binomial(1, self.sigma, edges.shape[0]).astype(bool)
                    | edges[:, -1].astype(bool),
                    :2,
                ]
                graph.remove_edges_from(edges_to_remove)

        # Add steady relationships and update the `is_single` status. Because we have
        # removed all casual relations above, we can simply use the degree as an
        # indicator of nodes being single.
        with timer("add_steady_edges"):
            nodes, num_partners = np.transpose(list(degree(graph)))
            candidates = nodes[
                np.random.uniform(size=nodes.shape)
                < np.where(num_partners, self.rho * (1 - self.xi), self.rho)
            ]
            add_edges_from_candidates(graph, candidates, type="steady")
        with timer("update_steady_node_status"):
            deg = list(degree(graph))
            graph._node.update(
                {
                    node: {"is_single": degree == 0, "has_casual": False}
                    for node, degree in deg
                }
            )

        # Add casual relationships. Because we have already removed all casual
        # relationships from the previous iteration, any edge is a steady relationship,
        # and the vertices of any edge are partnered up.
        with timer("add_casual_edges"):
            nodes, num_partners = np.transpose(deg)
            candidates = nodes[
                np.random.uniform(size=nodes.shape)
                < np.where(num_partners, self.omega1, self.omega0)
            ]
            edges = add_edges_from_candidates(graph, candidates, type="casual")
        with timer("update_casual_node_status"):
            for node in edges.ravel():
                graph._node[node]["has_casual"] = True

        return (graph, timer.times) if return_times else graph

    def evaluate_summaries(
        self, graph0: nx.Graph, graph1: nx.Graph
    ) -> dict[str, float]:
        steady_edges0 = {
            tuple(edge)
            for *edge, data in graph0.edges(data=True)
            if data["type"] == "steady"
        }
        steady_edges1 = {
            tuple(edge)
            for *edge, data in graph1.edges(data=True)
            if data["type"] == "steady"
        }

        # Evaluate number of nodes and nodes with casual relationships by relationship
        # status.
        num_nodes = {False: 0, True: 0}
        num_nodes_with_casual = {False: 0, True: 0}
        for graph in [graph0, graph1]:
            for _, data in graph.nodes(data=True):
                num_nodes[data["is_single"]] += 1
                num_nodes_with_casual[data["is_single"]] += data["has_casual"]

        return {
            "frac_retained_nodes": len(set(graph0) & set(graph1))
            / graph0.number_of_nodes(),
            "frac_retained_steady_edges": len(steady_edges0 & steady_edges1)
            / max(len(steady_edges0), 1),  # noqa: E131
            "frac_single_with_casual": num_nodes_with_casual[True]
            / max(num_nodes[True], 1),
            "frac_paired_with_casual": num_nodes_with_casual[False]
            / max(num_nodes[False], 1),
            "frac_paired": num_nodes[False]
            / (graph0.number_of_nodes() + graph1.number_of_nodes()),
            "num_steady_edges": len(steady_edges1),
            "num_casual_edges": sum(
                data["type"] == "casual" for *_, data in graph.edges(data=True)
            ),
            "num_nodes": graph.number_of_nodes(),
        }

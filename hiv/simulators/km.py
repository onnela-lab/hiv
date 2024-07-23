import networkx as nx
import numpy as np
from .util import Interval, Simulator, UnitInterval


class KretzschmarMorris(Simulator):
    r"""
    Simulate Kretzschmar and Morris model as described in
    https://doi.org/10.1016/0025-5564(95)00093-3.

    Args:
        n: Number of nodes.
        rho: Probability for two nodes to connect.
        sigma: Probability for a relation to dissolve.
        xi: Probability to reject a relationship if already in a relationship, i.e.,
            serial monogamy for :math:`\xi = 1` and random mixing for :math:`\xi = 0`.
    """

    arg_constraints = {
        "n": Interval(1, None, type=int),
        "sigma": UnitInterval(),
        "rho": UnitInterval(),
        "xi": UnitInterval(),
    }

    def __init__(self, n: int, sigma: float, rho: float, xi: float) -> None:
        self.n = n
        self.sigma = sigma
        self.rho = rho
        self.xi = xi
        self.validate_args()

    def init(self) -> nx.Graph:
        return nx.empty_graph(self.n)

    def step(self, graph: nx.Graph) -> nx.Graph:
        # Get candidates and their corresponding degrees. Make sure they are even.
        candidates_degrees = np.asarray(
            [item for item in graph.degree if np.random.binomial(1, self.rho)]
        )
        if len(candidates_degrees) % 2:
            candidates_degrees = candidates_degrees[1:]
        np.random.shuffle(candidates_degrees)
        edges, degrees = candidates_degrees.T.reshape(
            (2, len(candidates_degrees) // 2, 2)
        )
        proba = np.where(degrees.any(axis=-1), 1 - self.xi, 1)
        graph.add_edges_from(edges[np.random.uniform(size=proba.shape) < proba])

        # Dissolve existing edges.
        graph.remove_edges_from(
            edge for edge in graph.edges if np.random.binomial(1, self.sigma)
        )

        return graph

    def evaluate_summaries(
        self, graph0: nx.Graph, graph1: nx.Graph
    ) -> dict[str, float]:
        return {
            "frac_paired": (graph0.number_of_edges() + graph1.number_of_edges())
            / graph0.number_of_nodes()
        }

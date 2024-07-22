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
            serial monogamy for :math:`\xi = 1` and random mixing for
            :math:`\xi = 0`.
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
        # Iterate until a relation is formed.
        if np.random.uniform() < self.rho:
            while True:
                i, j = np.random.choice(graph.number_of_nodes(), size=2, replace=False)
                proba = 1 if max(graph.degree((i, j))) == 0 else 1 - self.xi
                if np.random.uniform() < proba:
                    graph.add_edge(i, j)
                    break

        # Pick an edge and possibly dissolve it.
        if graph.number_of_edges() and np.random.uniform() < self.sigma:
            edges = list(graph.edges)
            edge = edges[np.random.choice(len(edges))]
            graph.remove_edge(*edge)

        return graph

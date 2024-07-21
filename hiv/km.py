import networkx as nx
import numpy as np


def step(
    graph: nx.Graph,
    sigma: float,
    rho: float,
    xi: float,
) -> nx.Graph:
    r"""
    Simulate Kretzschmar and Morris model as described in
    https://doi.org/10.1016/0025-5564(95)00093-3.

    Args:
        graph: Graph to start the simulation from.
        rho: Probability for two nodes to connect.
        sigma: Probability for a relation to dissolve.
        xi: Probability to reject a relationship if already in a relationship, i.e.,
            serial monogamy for :math:`\xi = 1` and random mixing for :math:`\xi = 0`.
        num_steps: Number of steps to simulate.

    Returns:
        Evolved graph.
    """
    # Iterate until a relation is formed.
    if np.random.uniform() < rho:
        while True:
            i, j = np.random.choice(graph.number_of_nodes(), size=2, replace=False)
            proba = 1 if max(graph.degree((i, j))) == 0 else 1 - xi
            if np.random.uniform() < proba:
                graph.add_edge(i, j)
                break

    # Pick an edge and possibly dissolve it.
    if graph.number_of_edges() and np.random.uniform() < sigma:
        edges = list(graph.edges)
        edge = edges[np.random.choice(len(edges))]
        graph.remove_edge(*edge)

    return graph

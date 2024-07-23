import networkx as nx
import numpy as np
from .util import add_edges_from_candidates, Interval, Simulator, UnitInterval


class Stockholm(Simulator):
    """
    Simulate the Stockholm model, a discrete time version of the model defined in
    https://doi.org/10.1016/j.epidem.2019.02.001.

    Args:
        n: Expected number of nodes.
        mu: Probability for a node to emigrate and immigration rate `n * mu`.
        sigma: Probability for a steady relationship to dissolve.
        rho: Probability for a steady relationship to form between singles.
        w0: Probability for a casual relationship to form between singles.
        w1: Probability for a casual relationship to form between non-singles.
    """

    arg_constraints = {
        "n": Interval(0, None),
        "mu": UnitInterval(),
        "sigma": UnitInterval(),
        "rho": UnitInterval(),
        "w0": UnitInterval(),
        "w1": UnitInterval(),
    }

    def __init__(
        self, n: float, mu: float, sigma: float, rho: float, w0: float, w1: float
    ) -> None:
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.w0 = w0
        self.w1 = w1
        self.validate_args()

    def init(self) -> nx.Graph:
        return nx.empty_graph()

    def step(self, graph: nx.Graph) -> nx.Graph:
        label_offset = max(graph, default=-1) + 1

        # Remove nodes with the given migration probability. This may result in
        # `is_single` changing for nodes that are still in the graph. We update the
        # `is_single` flag below after removing steady relationships and adding new
        # ones.
        nodes_to_remove = [
            node
            for node, remove in zip(
                graph, np.random.binomial(1, self.mu, graph.number_of_nodes())
            )
            if remove
        ]
        graph.remove_nodes_from(nodes_to_remove)

        # Add new nodes that have migrated in.
        num_new_nodes = np.random.poisson(self.n * self.mu)
        graph.add_nodes_from(label_offset + np.arange(num_new_nodes))

        # Remove steady relationships with probability sigma and all casual
        # relationships.
        edges_to_remove = [
            edge
            for *edge, data in graph.edges(data=True)
            if data["is_casual"] or np.random.binomial(1, self.sigma)
        ]
        graph.remove_edges_from(edges_to_remove)

        # Add steady relationships and update the `is_single` status. Because we have
        # removed all casual relations above, we can simply use the degree as an
        # indicator of nodes being single.
        add_edges_from_candidates(
            graph,
            [
                node
                for node, degree in graph.degree
                if degree == 0 and np.random.binomial(1, self.rho)
            ],
            is_casual=False,
        )
        graph._node.update(
            {
                node: {"is_single": degree == 0, "has_casual": False}
                for node, degree in graph.degree
            }
        )

        # Add casual relationships. Because we have already removed all casual
        # relationships from the previous iteration, any edge is a steady relationship,
        # and the vertices of any edge are partnered up.
        candidates = [
            node
            for node, data in graph.nodes(data=True)
            if np.random.binomial(1, self.w0 if data["is_single"] else self.w1)
        ]
        edges = add_edges_from_candidates(graph, candidates, is_casual=True)
        for node in edges.ravel():
            graph._node[node]["has_casual"] = True

        return graph

    def evaluate_summaries(
        self, graph0: nx.Graph, graph1: nx.Graph
    ) -> dict[str, float]:
        steady_edges0 = {
            tuple(edge)
            for *edge, data in graph0.edges(data=True)
            if not data["is_casual"]
        }
        steady_edges1 = {
            tuple(edge)
            for *edge, data in graph1.edges(data=True)
            if not data["is_casual"]
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
        }

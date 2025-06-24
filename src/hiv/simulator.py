import collectiontools
import numpy as np
from typing import overload, Literal
from .util import candidates_to_edges, NumpyGraph, Timer, decompress_edges


empty_int_array = np.asarray((), dtype=np.uint64)


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
        xi: Propensity for concurrency.
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

    def run(
        self, graph: NumpyGraph, num_steps: int, validate: bool = False
    ) -> NumpyGraph:
        for _ in range(num_steps):
            graph = self.step(graph)
            if validate:
                graph.validate()
        return graph

    def validate_args(self) -> None:
        for arg, constraint in self.arg_constraints.items():
            if not constraint.is_valid(getattr(self, arg)):
                raise ValueError(
                    f"Argument `{arg}` does not satisfy constraint `{constraint}`."
                )

    def init(self) -> NumpyGraph:
        n = round(self.n)
        return NumpyGraph(
            nodes=np.arange(n),
            node_attrs={"last_casual_at": -np.ones(n, dtype=int)},
            edge_attrs={"steady": bool, "created_at": int},
            attrs={"step": 0},
        )

    @overload
    def step(
        self, graph: NumpyGraph, return_times: Literal[False] = False
    ) -> NumpyGraph: ...

    @overload
    def step(
        self, graph: NumpyGraph, return_times: Literal[True]
    ) -> tuple[NumpyGraph, dict[str, float]]: ...

    def step(
        self, graph: NumpyGraph, return_times: bool = False
    ) -> NumpyGraph | tuple[NumpyGraph, dict[str, float]]:
        label_offset = (graph.nodes.max() + 1) if graph.nodes.size else 0
        timer = Timer()

        # Keep nodes with probability 1 - mu.
        with timer("remove_nodes"):
            graph.filter_nodes(np.random.uniform(size=graph.nodes.size) > self.mu)

        # Add new nodes that have migrated in.
        with timer("add_nodes"):
            num_new_nodes = np.random.poisson(self.n * self.mu)
            graph.add_nodes(label_offset + np.arange(num_new_nodes), last_casual_at=-1)

        # If there are no nodes, there's nothing else to be done.
        if not graph.nodes.size:
            return graph

        # Remove steady relationships with probability sigma and remove all casual
        # edges with probability 1. When the probability to remove is high, we want the
        # corresponding filter to be False.
        with timer("remove_edges"):
            proba_remove = np.where(graph.edge_attrs["steady"], self.sigma, 1)
            fltr = proba_remove < np.random.uniform(size=proba_remove.size)
            graph.filter_edges(fltr)

        with timer("add_steady_edges"):
            steady_degrees = graph.degrees(key=lambda attrs: attrs["steady"])
            is_partnered = steady_degrees > 0
            proba = np.where(is_partnered, self.rho * self.xi, self.rho)
            fltr = np.random.uniform(size=graph.nodes.size) < proba
            candidates = graph.nodes[fltr]

            new_steady_edges = candidates_to_edges(candidates)
            # Deduplicate the edges because two people may form the same relationship
            # again if we allow concurrency, they both seek a new relationship, and they
            # then get paired up with each other. An edge case, but happens in large
            # simulations.
            if self.xi:
                new_steady_edges = np.setdiff1d(
                    new_steady_edges, graph.edges[graph.edge_attrs["steady"]]
                )
            graph.add_edges(
                new_steady_edges, steady=True, created_at=graph.attrs["step"]
            )

        # Add casual relationships. Because we have already removed all casual
        # relationships from the previous iteration, any edge is a steady relationship,
        # and the vertices of any edge are partnered up.
        with timer("add_casual_edges"):
            # Seek casual edges with probability depending on being single. We don't
            # need to check if there are steady edges because we just created them.
            newly_partnered = np.isin(graph.nodes, decompress_edges(new_steady_edges))
            is_partnered |= newly_partnered
            proba = np.where(is_partnered, self.omega1, self.omega0)
            fltr = np.random.uniform(size=graph.nodes.size) < proba
            candidates = graph.nodes[fltr]

            # Add new edges and exclude edges that already exist in the steady edge set.
            # We need to first compress the edges to allow the set operations.
            casual_edges = candidates_to_edges(candidates)
            casual_edges = np.setdiff1d(casual_edges, graph.edges)
            graph.add_edges(casual_edges, steady=False, created_at=graph.attrs["step"])

            # Update the time since the last casual encounter.
            fltr = np.isin(graph.nodes, decompress_edges(casual_edges))
            graph.node_attrs["last_casual_at"][fltr] = graph.attrs["step"]

        graph.attrs["step"] += 1

        return (graph, timer.times) if return_times else graph

    def evaluate_pointwise_summaries(
        self, graph: NumpyGraph, sample: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate pointwise summary statistics for a single graph.

        Args:
            graph: Graph for which to evaluate summaries.
            sample: Sample of nodes to evaluate summaries.

        Returns:
            Summary statistics.
        """
        assert np.setdiff1d(sample, graph.nodes).size == 0
        # Get the steady degree of individuals who are in the sample. Their partner may
        # be outside of the sample, however.
        sample_has_node = np.isin(graph.nodes, sample)
        steady_degrees = graph.degrees(key=lambda attrs: attrs["steady"])[
            sample_has_node
        ]
        assert steady_degrees.shape == sample.shape

        # Get the frequency of different steady degrees.
        num_nodes_by_steady_degree = np.bincount(steady_degrees, minlength=1)
        assert num_nodes_by_steady_degree.sum() == sample.size

        # Get a binary indicator if a node has a casual contact.
        has_casual = np.isin(
            sample, decompress_edges(graph.edges[~graph.edge_attrs["steady"]])
        )
        assert has_casual.shape == sample.shape

        # Get the degree frequency for steady connections where the node also has a
        # casual contact.
        num_nodes_by_steady_degree_with_casual = np.bincount(
            steady_degrees[has_casual], minlength=1
        )
        assert num_nodes_by_steady_degree_with_casual.sum() <= sample.size

        # Get all steady edges so we can compare how many were retained.
        sample_has_edge = np.isin(decompress_edges(graph.edges), sample).any(axis=-1)
        sample_has_edge_and_is_steady = graph.edge_attrs["steady"] & sample_has_edge
        steady_edges = graph.edges[sample_has_edge_and_is_steady]

        # Evaluate steady relationship durations clipped above at 52 weeks as reported
        # by Hansson et al. (2019). The clipping occurs because of the collection
        # method: "showing the participant a timeline over the last 12 months where
        # participants added the time period for a sexual relationship with a sex
        # partner."
        if steady_edges.size:
            steady_length: np.ndarray = (
                graph.attrs["step"]
                - graph.edge_attrs["created_at"][sample_has_edge_and_is_steady]
            )
            # We divide by 52 to get summaries on the same scale as the `frac_*`.
            steady_length = steady_length.clip(max=52).mean() / 52
        else:
            # If there are no steady edges, we set the relationship duration to zero,
            # consistent with there being no relationships.
            steady_length = 0

        # FIXME: This is actually the time *since* the last contact, not the gap between
        # contacts.
        # Evaluate the gap since the last casual sexual contact as reported in Hansson
        # et al. (2019). We only consider gaps where ALL of the following apply:
        #
        # - `in_sample` is true (because we won't observe otherwise).
        # - `last_casual_at` is not -1 (that indicates no casual contact yet)
        # - `casual_gap` is no more than 52 weeks because those casual contacts would
        #   not have come up in the questionnaire that focused on the past year.
        #
        # If there are no casual contacts, we set the gap to the largest value
        # consistent with there being no casual contacts.
        #
        # We process the data in stages to ensure we can break down by being in a steady
        # relationship.
        last_casual_at = graph.node_attrs["last_casual_at"][sample_has_node]
        has_previous_casual = last_casual_at != -1
        last_casual_at = last_casual_at[has_previous_casual]
        has_partner = steady_degrees[has_previous_casual] > 0

        casual_gap_single = last_casual_at[~has_partner]
        if casual_gap_single.size:
            casual_gap_single = casual_gap_single.mean() / 52
        else:
            casual_gap_single = 1

        casual_gap_paired = last_casual_at[has_partner]
        if casual_gap_paired.size:
            casual_gap_paired = casual_gap_paired.mean() / 52
        else:
            casual_gap_paired = 1

        return {
            # Fraction of nodes who have at least one steady relationship.
            "frac_paired": num_nodes_by_steady_degree[1:].sum() / max(sample.size, 1),
            # Fraction of nodes with at least one steady relationship who have more than
            # one steady relationship.
            "frac_concurrent": num_nodes_by_steady_degree[2:].sum()
            / max(num_nodes_by_steady_degree[1:].sum(), 1),
            # Fraction of singles who have a casual contact.
            "frac_single_with_casual": num_nodes_by_steady_degree_with_casual[0]
            / max(num_nodes_by_steady_degree[0], 1),
            # Fraction of paired individuals who have a casual contact.
            "frac_paired_with_casual": num_nodes_by_steady_degree_with_casual[1:].sum()
            / max(num_nodes_by_steady_degree[1:].sum(), 1),
            # Steady edges where at least one member is in the sample which we'll use to
            # evaluate the fraction of retained steady edges. We prefix with "_" because
            # we don't actually want to save these statistics.
            "_steady_edges": steady_edges,
            # Summaries from Hansson et al. (2019).
            "steady_length": steady_length,
            "casual_gap_single": casual_gap_single,
            "casual_gap_paired": casual_gap_paired,
        }

    def evaluate_longitudinal_summaries(
        self, graph0: NumpyGraph, graph1: NumpyGraph, sample0: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Evaluate (longitudinal) summary statistics for two graphs.

        Args:
            graph0: Graph at first observation.
            graph1: Graph at second observation.
            sample0: Nodes to include in the initial sample.

        Returns:
            Dictionary of summary statistics.
        """
        # Subsample the graphs.
        if sample0 is None:
            sample0 = graph0.nodes
        sample1 = np.intersect1d(sample0, graph1.nodes)

        # Evaluate pointwise summaries.
        pointwise: dict[str, float] = {}
        collectiontools.append_values(
            pointwise, self.evaluate_pointwise_summaries(graph0, sample0)
        )
        collectiontools.append_values(
            pointwise, self.evaluate_pointwise_summaries(graph1, sample1)
        )

        counts = np.asarray([sample0.size, sample1.size])
        weights = counts / counts.sum()

        # Purely longitudinal statistics.
        summaries = {
            # Fraction of nodes that are retained in the sample.
            "frac_retained_nodes": np.intersect1d(sample0, sample1).size
            / max(sample0.size, 1),
            # Fraction of edges that have at least one vertex in the sample.
            "frac_retained_steady_edges": np.intersect1d(
                *pointwise["_steady_edges"]
            ).size
            / max(pointwise["_steady_edges"][0].size, 1),
        }

        # Cross-sectional statistics averaged over the two samples, weighted by the
        # sample size.
        pointwise = {
            key: value for key, value in pointwise.items() if not key.startswith("_")
        }
        summaries.update(
            {key: np.dot(weights, value) for key, value in pointwise.items()}
        )

        # Cross sectional statistics where we keep track separately, e.g., for
        # debugging.
        summaries.update({f"_{key}": value for key, value in pointwise.items()})
        return summaries


def estimate_paired_fraction(
    rho: np.ndarray, mu: np.ndarray, sigma: np.ndarray, lag: np.ndarray | None = None
) -> np.ndarray:
    """
    Estimate the fraction of paired nodes.

    Args:
        rho: Probability to seek a new relationship.
        mu: Probability to leave the population.
        sigma: Probability for a relationship to dissolve naturally.
        lag: Time since first interviewing the cohort. This is important because
            subsequent interviews are with people who have, by definition, not left the
            population, which leads to lower break-up rates and higher fraction of
            paired nodes.

    Returns:
        Expected fraction of paired nodes.
    """
    beta = (1 - mu) * (1 - sigma) * (1 - rho)
    alpha = beta * (1 - mu)

    if lag is None:
        return rho / (1 - alpha)

    return (rho / (1 - beta)) * (1 - (beta ** (lag + 1)) * (mu / (1 - alpha)))

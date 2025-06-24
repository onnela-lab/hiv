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
        return NumpyGraph(
            nodes=np.arange(round(self.n)),
            edge_attributes={"steady": bool, "created_at": int},
            attributes={"step": 0},
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
            graph.add_nodes(label_offset + np.arange(num_new_nodes))

        # If there are no nodes, there's nothing else to be done.
        if not graph.nodes.size:
            return graph

        # Remove steady relationships with probability sigma and remove all casual
        # edges with probability 1. When the probability to remove is high, we want the
        # corresponding filter to be False.
        with timer("remove_edges"):
            proba_remove = np.where(graph.edge_attributes["steady"], self.sigma, 1)
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
                    new_steady_edges, graph.edges[graph.edge_attributes["steady"]]
                )
            graph.add_edges(
                new_steady_edges, steady=True, created_at=graph.attributes["step"]
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
            graph.add_edges(
                casual_edges, steady=False, created_at=graph.attributes["step"]
            )
        graph.attributes["step"] += 1

        return (graph, timer.times) if return_times else graph

    def evaluate_summaries(
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
        # Construct the set of nodes that we will use for each of the two graphs.
        if sample0 is None:
            sample0 = graph0.nodes
        else:
            assert np.isin(
                sample0, graph0.nodes
            ).all(), "Some nodes in initial `sample0` are not present in `graph0`."
        samples: list[np.ndarray] = [sample0, np.intersect1d(sample0, graph1.nodes)]

        # Evaluate summary statistics for each of the graphs. We will then consolidate
        # these statistics to obtain summaries for approximate Bayesian computation.
        summaries: dict[str, list[float | int | np.ndarray]] = {}
        for graph, sample in zip([graph0, graph1], samples):
            # Get steady degrees of nodes and degree distribution.
            degrees = graph.degrees(key=lambda attrs: attrs["steady"])
            assert degrees.shape == graph.nodes.shape, (
                f"Expected degree vector shape ({degrees.shape}) to match nodes shape "
                f"({graph.nodes.shape})."
            )
            degrees = degrees[np.isin(graph.nodes, sample)]
            assert degrees.size == sample.size, (
                f"Expected degree vector shape ({degrees.shape}) to match node sample "
                f"shape ({sample.shape})."
            )
            num_nodes_by_degree = np.bincount(degrees)
            assert num_nodes_by_degree.sum() <= sample.size, (
                f"Expected sum of degree distribution ({num_nodes_by_degree.sum()}) "
                f"to match sample size ({sample.size})."
            )

            # Filter to nodes that had a casual interaction.
            has_casual = np.isin(
                sample, decompress_edges(graph.edges[~graph.edge_attributes["steady"]])
            )
            assert has_casual.shape == sample.shape, (
                f"Expected the `has_casual` indicator shape ({has_casual.shape}) to "
                f"match the sample shape ({sample.shape})."
            )
            degrees_with_casual = degrees[has_casual]
            assert degrees_with_casual.shape == (has_casual.sum(),), (
                f"Expected the degree vector shape ({degrees_with_casual.shape}) for "
                "nodes with casual partners to match the number of nodes with casual "
                f"partners ({has_casual.sum()})."
            )
            num_nodes_by_degree_with_casual = np.bincount(
                degrees_with_casual, minlength=1
            )

            # Get compressed steady edges so we can evaluate how many are retained. We
            # restrict to edges where at least one member is in the sample.
            steady_edges = graph.edges[graph.edge_attributes["steady"]]
            steady_edges = steady_edges[
                np.isin(decompress_edges(steady_edges), sample).any(axis=1)
            ]

            # Append values to the summaries. Any keys prefixed with _ will not be
            # stored. E.g., the vector of steady edges may be large, leading to big
            # files (~GB) for the simulations.
            collectiontools.append_values(
                summaries,
                {
                    "sample_size": sample.size,
                    "num_nodes_by_degree": num_nodes_by_degree,
                    "num_nodes_by_degree_with_casual": num_nodes_by_degree_with_casual,
                    "_steady_edges": steady_edges,
                    "frac_paired": num_nodes_by_degree[1:].sum() / max(sample.size, 1),
                    "frac_single_with_casual": num_nodes_by_degree_with_casual[0]
                    / max(num_nodes_by_degree[0], 1),
                    "frac_paired_with_casual": num_nodes_by_degree_with_casual[1:].sum()
                    / max(num_nodes_by_degree[1:].sum(), 1),
                    "min_node_label": sample.min() if sample.size else -1,
                    "max_node_label": sample.max() if sample.size else -1,
                    # Clipped length of steady relationships from Hansson et al. (2019).
                    # We divide by 52 to get another summary on the [0, 1] scale.
                    "steady_length": (
                        (
                            graph.attributes["step"]
                            - graph.edge_attributes["created_at"][
                                graph.edge_attributes["steady"]
                            ]
                        )
                        .clip(max=52)
                        .mean()
                        / 52
                        if graph.edge_attributes["steady"].any()
                        else 0
                    ),
                },
            )

        weight = np.asarray(summaries["sample_size"]) / max(
            sum(summaries["sample_size"]), 1  # type: ignore
        )

        # Summary statistics that are of particular interest to us for longitudinal
        # surveys.
        result = {
            # Fraction of nodes retained which is monotonically decreasing. We expect
            # this statistics to change slowly because the migration probability `mu`
            # tends to be small.
            "frac_retained_nodes": np.intersect1d(*samples, assume_unique=True).size
            / max(summaries["sample_size"][0], 1),  # type: ignore
            # Fraction of retained steady edges. This is generally decreasing but there
            # may be increasing "blips" because a relationship could re-form, and we
            # don't ask "did you break up and make up again" in the survey. This
            # statistic is indicative of the break up probability `sigma` and to a
            # lesser extent the migration probability `mu`.
            "frac_retained_steady_edges": np.intersect1d(
                *summaries["_steady_edges"], assume_unique=True
            ).size
            / max(summaries["_steady_edges"][0].size, 1),  # type: ignore
            # Fraction of singles with a casual contact. This statistics is informative
            # of `omega_0`.
            "frac_single_with_casual": np.dot(
                summaries["frac_single_with_casual"], weight  # type: ignore
            ),
            # Fraction of individuals in steady relationships with a casual contact.
            # This statistics is indicative of `omega_1`.
            "frac_paired_with_casual": np.dot(
                summaries["frac_paired_with_casual"], weight  # type: ignore
            ),
            # Fraction of nodes that have one or more steady relations. This statistics
            # informs `rho`, the tendency to form connections. In contrast to other
            # statistics, such as the fraction of retained nodes, this statistic is
            # also affected by parameters like the dissolution rate `sigma`, emigration
            # rate `mu`, and concurrency parameter `xi`.
            "frac_paired": np.dot(summaries["frac_paired"], weight),  # type: ignore
            # Fraction of nodes in a steady relationship who have more than one steady
            # relationship. This is indicative of the monogamy parameter `xi`.
            "frac_concurrent": num_nodes_by_degree[2:].sum()  # type: ignore
            / max(num_nodes_by_degree[1:].sum(), 1),  # type: ignore
        }

        # Summary statistics from Hansson et al. (2019) from a survey on MSM in Sweden.
        result.update(
            {
                # Fraction of paired individuals (reported as 0.64) is already covered
                # above.
                # They report the mean time between casual sexual partners as 101.9 days if
                # in a relationship and 62.6 days if not in a relationship. We can translate
                # that into the fraction of singles/paired with a casual partner in the last
                # week. 0.06639 for paired, 0.1058 for singles.
                # Fraction of concurrent relationships is 0.26 from S2.3 of the supplement.
                # Any "frac_retained", we cannot get our hands on because the survey has
                # only a single wave. Constraining `mu` is fundamentally difficult, but we
                # can use something like the typical length of a relationship as additional
                # information. They note that the self-reported mean duration of a
                # partnership is 203.2 days = 29.03 weeks, although that is capped at one
                # year or about 52 weeks.
                "steady_length": np.dot(summaries["steady_length"], weight),
            }
        )

        # Debugging summary statistics.
        result.update(
            {
                f"_{key}": value
                for key, value in summaries.items()
                if not key.startswith("_")
            }
        )
        return result


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

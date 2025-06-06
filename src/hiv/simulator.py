import collectiontools
import numpy as np
from .util import candidates_to_edges, NumpyGraph, Timer, decompress_edges


empty_int_array = np.asarray((), dtype=np.uint64)


def add_padded(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(a.size, b.size)
    return np.pad(a, (0, n - a.size)) + np.pad(b, (0, n - b.size))


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
        return NumpyGraph(np.arange(round(self.n)))

    def step(self, graph: NumpyGraph, return_times: bool = False) -> NumpyGraph:
        label_offset = (graph.nodes.max() + 1) if graph.nodes.size else 0
        timer = Timer()

        # Keep nodes with probability 1 - mu.
        with timer("remove_nodes"):
            graph.nodes = graph.nodes[
                np.random.uniform(size=graph.nodes.size) > self.mu
            ]

        # Remove edges incident on a removed node.
        if "steady" in graph.edges:
            with timer("remove_lingering_edges"):
                compressed_steady = graph.edges["steady"]
                steady = decompress_edges(compressed_steady)
                # We can't use assume_unique here because node indices may appear
                # repeatedly in the edge list.
                graph.edges["steady"] = compressed_steady[
                    np.isin(steady, graph.nodes).all(axis=-1)
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

        # Keep steady relationships with probability 1 - sigma.
        if "steady" in graph.edges:
            with timer("remove_steady_edges"):
                steady = graph.edges["steady"]
                graph.edges["steady"] = steady[
                    np.random.uniform(size=steady.size) > self.sigma
                ]

        with timer("add_steady_edges"):
            # Seek steady edges with probability depending on being single.
            if "steady" in graph.edges:
                # We can't use assume_unique here because node indices may appear
                # repeatedly in the edge list.
                degrees = graph.degrees("steady")
                is_partnered = degrees > 0
                # proba = np.where(is_partnered, self.rho * self.xi, self.rho)
                proba = self.rho * self.xi**degrees
            else:
                is_partnered = None
                proba = self.rho
            fltr = np.random.uniform(size=graph.nodes.size) < proba
            candidates = graph.nodes[fltr]

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
            newly_partnered = np.isin(graph.nodes, decompress_edges(new_steady_edges))
            if is_partnered is None:
                is_partnered = newly_partnered
            else:
                is_partnered |= newly_partnered
            proba = np.where(is_partnered, self.omega1, self.omega0)
            fltr = np.random.uniform(size=graph.nodes.size) < proba
            candidates = graph.nodes[fltr]

            # Add new edges and exclude edges that already exist in the steady edge set.
            # We need to first compress the edges to allow the set operations.
            casual_edges = candidates_to_edges(candidates)
            graph.edges["casual"] = np.setdiff1d(
                casual_edges, graph.edges["steady"], assume_unique=True
            )

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
            # Get degrees of nodes and degree distribution.
            degrees = graph.degrees("steady")
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
                sample, decompress_edges(graph.edges.get("casual", empty_int_array))
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
            steady_edges = graph.edges.get("steady", empty_int_array)
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
                    / max(num_nodes_by_degree_with_casual[0], 1),
                    "frac_paired_with_casual": num_nodes_by_degree_with_casual[1:].sum()
                    / max(num_nodes_by_degree_with_casual[1:].sum(), 1),
                    "min_node_label": sample.min() if sample.size else -1,
                    "max_node_label": sample.max() if sample.size else -1,
                },
            )

        weight = np.asarray(summaries["sample_size"]) / max(
            sum(summaries["sample_size"]), 1
        )
        return {
            # Fraction of nodes retained which is monotonically decreasing. We expect
            # this statistics to change slowly because the migration probability `mu`
            # tends to be small.
            "frac_retained_nodes": np.intersect1d(*samples, assume_unique=True).size
            / max(summaries["sample_size"][0], 1),
            # Fraction of retained steady edges. This is generally decreasing but there
            # may be increasing "blips" because a relationship could re-form, and we
            # don't ask "did you break up and make up again" in the survey. This
            # statistic is indicative of the break up probability `sigma`.
            "frac_retained_steady_edges": np.intersect1d(
                *summaries["_steady_edges"], assume_unique=True
            ).size
            / max(summaries["_steady_edges"][0].size, 1),
            # Fraction of singles with a casual contact. This statistics is informative
            # of `omega0`.
            "frac_single_with_casual": np.dot(
                summaries["frac_single_with_casual"], weight
            ),
            # Fraction of individuals in steady relationships with a casual contact.
            # This statistics is indicative of `omega1`.
            "frac_paired_with_casual": np.dot(
                summaries["frac_paired_with_casual"], weight
            ),
            # Fraction of nodes that have one or more steady relations. This statistics
            # informs `rho`, the tendency to form connections. In contrast to other
            # statistics, such as the fraction of retained nodes, this statistic is
            # also affected by parameters like the dissolution rate `sigma` and
            # emigration rate `mu`.
            "frac_paired": np.dot(summaries["frac_paired"], weight),
            # Fraction of nodes in a steady relationship who have more than one steady
            # relationship. This is indicative of the monogamy parameter `xi`.
            "frac_concurrent": num_nodes_by_degree[2:].sum()
            / max(num_nodes_by_degree[1:].sum(), 1),
        } | {
            f"_{key}": value
            for key, value in summaries.items()
            if not key.startswith("_")
        }


def estimate_paired_fraction(rho, mu, sigma):
    """
    Estimate the fraction of paired nodes.

    This differs from the continuous-time version based on inline text about 3/4 of the
    way down page 369 of 10.1016/j.idm.2017.07.002: rho / (rho + sigma + 2 * mu)
    """
    return rho / (1 - (1 - mu) ** 2 * (1 - sigma) * (1 - rho))

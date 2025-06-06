import argparse
import collectiontools
from datetime import datetime
import numbers
import numpy as np
from pathlib import Path
import pickle
from scipy import stats
from ..simulator import UniversalSimulator
from ..util import assert_graphs_equal, decompress_edges, to_np_dict
from tqdm import tqdm


prior_clss = {"beta": stats.beta}

# Default preset with sensible prior parameters informed by the literature. See the
# parameter-normalization and prior notebooks for details.
default_preset = {
    "n": 5_000,
    "mu": stats.beta(2.7471745380604844, 2162.302624350188),
    "rho": stats.beta(1.898530545859645, 24.62575083539298),
    "sigma": stats.beta(2.0596220554513454, 98.67884889543724),
    "omega0": stats.beta(3.82125145332353, 5.622913762790063),
    "omega1": stats.beta(9.308666573257643, 47.55873620969334),
    "xi": stats.beta(2.0, 2.0),
}

prior_presets: dict[str, dict[str, stats.distributions.rv_continuous]] = {
    "empty": {},
    "default": default_preset,
    # Discrete-time stochastic simulator based on the continuous-time stochastic
    # simulator of Hansson et al. (2019) and the continous-time deterministic simulator
    # of Xiridou et al. (2003) obtained by setting :math:`\xi = 0`, i.e., serial
    # monogamy.
    "hansson2019": default_preset | {"xi": 0},
    # Discrete-time stochastic simulator based on the stochastic discrete-time simulator
    # of Kretzschmar et al. (1996) obtained by setting :math:`\mu = 0` and
    # :math:`w_0 = w_1 = 0`, i.e., a closed population without casual contacts.
    "kretzschmar1996": default_preset | {"omega0": 0, "omega1": 0, "mu": 0},
    # Discrete-time stochastic simulator based on the deterministic continuous-time
    # simulator of Leng et al. (2018) obtained by setting :math:`\mu = 0` and
    # :math:`\xi = 0`, i.e., a closed population and serial monogamy.
    "leng2018": default_preset | {"xi": 0, "mu": 0},
    # Discrete-time stochastic simulator based on Kretzschmar et al. (1998) obtained by
    # setting :math:`w_0 = w_1 = 0`, i.e., no casual sexual partners. They use a
    # different parameterization of the immigration rate :math:\nu = n / \mu` instead of
    # expected population size like we do here.
    "kretzschmar1998": default_preset | {"omega0": 0, "omega1": 0},
}


def parse_prior(
    param: str,
) -> tuple[str, numbers.Number | stats.distributions.rv_continuous]:
    """
    Parse a prior specification.
    """
    parts = param.split("=")
    assert (
        len(parts) == 2
    ), f"Parameter specification `{param}` does not have format `[name]=[spec]`."
    arg, spec = parts

    if ":" in spec:
        cls_name, hyperparams = spec.split(":")
        prior = prior_clss[cls_name](*map(float, hyperparams.split(",")))
    elif "." in spec:
        prior = float(spec)
    else:
        prior = int(spec)
    return arg, prior


def parse_priors(params):
    """
    Parse priors into a dictionary.
    """
    return dict(map(parse_prior, params))


class Args:
    preset: str
    num_samples: int
    num_lags: int
    output: Path
    burnin: int
    param: list[str]
    save_graphs: bool
    seed: int
    sample_size: int | None


def __main__(argv=None) -> None:
    if argv:
        argv = list(map(str, argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="random number generator seed")
    parser.add_argument(
        "--burnin",
        help="number of burn in steps (defaults to the expected population size `n`)",
        type=int,
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="store graph sequences in addition to summaries (memory intensive)",
    )
    parser.add_argument(
        "--param",
        "-p",
        help="parameter value as [name]=[value] or prior as [name]=[prior_cls]:[*args]",
        action="append",
    )
    parser.add_argument(
        "--preset", help="prior preset to use", choices=prior_presets, default="default"
    )
    parser.add_argument("--sample-size", help="size of survey sample", type=int)
    parser.add_argument("num_samples", help="number of samples", type=int)
    parser.add_argument("num_lags", help="number of lags to consider", type=int)
    parser.add_argument("output", help="output path for the samples", type=Path)
    args: Args = parser.parse_args(argv)

    priors = prior_presets[args.preset].copy()
    priors.update(parse_priors(args.param or []))
    extra = set(priors) - set(UniversalSimulator.arg_constraints)
    assert not extra, (
        f"Parameters {extra} are not allowed; allowed parameters are "
        f"{UniversalSimulator.arg_constraints}."
    )

    if args.seed is not None:
        np.random.seed(args.seed)

    # Container for results.
    result = {
        "args": vars(args),
        "priors": priors,
        "start": datetime.now(),
    }
    # Generate parameter values up front. This means that for fixed seed, we always have
    # the same parameter values even if the burnin is different, etc.
    result["params"] = {
        arg: (
            prior * np.ones(args.num_samples)
            if isinstance(prior, numbers.Number)
            else prior.rvs(args.num_samples)
        )
        for arg, prior in priors.items()
    }

    for i in tqdm(range(args.num_samples)):
        # Sample and use fixed values if provided.
        params = {key: value[i] for key, value in result["params"].items()}
        simulator = UniversalSimulator(**params)

        # Run the burnin to get the first sample and validate the graph.
        burnin = int(params["n"]) if args.burnin is None else args.burnin
        graph0 = simulator.init()
        graph0 = simulator.run(graph0, burnin)
        graph0.validate()
        graph1 = graph0.copy()

        # Initialize graph sequences.
        if args.save_graphs:
            graph_sequence = []
            result.setdefault("graph_sequences", []).append(graph_sequence)
        else:
            graph_sequence = None

        # Initialize summary sequences.
        summaries = {}

        if args.sample_size:
            initial_sample = np.random.choice(
                graph0.nodes, args.sample_size, replace=False
            )
        else:
            initial_sample = graph0.nodes

        for step in range(args.num_lags):
            if args.save_graphs:
                graph_sequence.append(graph1.copy())
            lag_summaries = simulator.evaluate_summaries(graph0, graph1, initial_sample)

            # Sanity check for summary statistics if no evolution has happened yet.
            if step == 0:
                assert_graphs_equal(graph0, graph1)
                assert (
                    lag_summaries["frac_retained_nodes"] == 1
                    or initial_sample.size == 0
                ), (
                    "Fraction of retained nodes is "
                    f"{lag_summaries['frac_retained_nodes']}, but the graph is "
                    "unchanged."
                )
                steady_edges = graph0.edges["steady"]
                steady_edges = steady_edges[
                    np.isin(decompress_edges(steady_edges), initial_sample).any(axis=1)
                ]
                assert (
                    lag_summaries["frac_retained_steady_edges"] == 1
                    or steady_edges.size == 0
                ), (
                    "Fraction of retained edges is "
                    f"{lag_summaries['frac_retained_steady_edges']}, but the graph is "
                    "unchanged."
                )
            collectiontools.append_values(summaries, lag_summaries)
            # Skip simulation for the last step.
            if step == args.num_lags - 1:
                continue
            simulator.step(graph1)

        # Validate that the final graph is still valid.
        graph1.validate()

        # Add summaries to the sequence.
        collectiontools.append_values(result.setdefault("summaries", {}), summaries)

    end = datetime.now()
    result.update(
        {
            "params": to_np_dict(result["params"]),
            "summaries": to_np_dict(
                result["summaries"], lambda key: not key.startswith("_")
            ),
            "end": end,
            "duration": (end - result["start"]).total_seconds(),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()

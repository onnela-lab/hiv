import argparse
import collectiontools
import numbers
import numpy as np
from pathlib import Path
import pickle
from scipy import stats
from ..simulator import UniversalSimulator
from ..util import to_np_dict
from tqdm import tqdm


prior_clss = {"beta": stats.beta}

# Default preset with sensible prior parameters informed by the literature. See the
# parameter-normalization and prior notebooks for details.
default_preset = {
    "n": 5_000,
    "mu": stats.beta(2.1465, 1301.0494),
    "rho": stats.beta(1.125, 7.9159),
    "sigma": stats.beta(1.0593, 32.5972),
    "omega0": stats.beta(2.579, 4.471),
    "omega1": stats.beta(2.8184, 13.9943),
    "xi": stats.beta(2, 2),
}

prior_presets = {
    "empty": {},
    "default": default_preset,
    # Discrete-time stochastic simulator based on the continuous-time stochastic
    # simulator of Hansson et al. (2019) and the continous-time deterministic simulator
    # of Xiridou et al. (2003) obtained by setting :math:`\xi = 1`, i.e., serial
    # monogamy.
    "hansson2019": default_preset | {"xi": 1.0},
    # Discrete-time stochastic simulator based on the stochastic discrete-time simulator
    # of Kretzschmar et al. (1996) obtained by setting :math:`\mu = 0` and
    # :math:`w_0 = w_1 = 0`, i.e., a closed population without casual contacts.
    "kretzschmar1996": default_preset | {"omega0": 0, "omega1": 0, "mu": 0},
    # Discrete-time stochastic simulator based on the deterministic continuous-time
    # simulator of Leng et al. (2018) obtained by setting :math:`\mu = 0` and
    # :math:`\xi = 1`, i.e., a closed population and serial monogamy.
    "leng2018": default_preset | {"xi": 0, "mu": 0},
    # Discrete-time stochastic simulator based on Kretzschmar et al. (1998) obtained by
    # setting :math:`w_0 = w_1 = 0`, i.e., no casual sexual partners. They use a
    # different parameterization of the immigration rate :math:\nu = n / \mu` instead of
    # expected population size like we do here.
    "kretzschmar1998": default_preset | {"omega0": 0, "omega1": 0},
}


def parse_prior(param):
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


def __main__(argv=None) -> None:
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

    result = {
        "args": vars(args),
        "priors": priors,
    }
    for _ in tqdm(range(args.num_samples)):
        # Sample and use fixed values if provided.
        params = {
            arg: prior if isinstance(prior, numbers.Number) else prior.rvs()
            for arg, prior in priors.items()
        }
        simulator = UniversalSimulator(**params)

        # Run the burnin to get the first sample.
        burnin = int(params["n"]) if args.burnin is None else args.burnin
        graph0 = simulator.init()
        graph0 = simulator.run(graph0, burnin)
        graph1 = graph0.copy()

        # Initialize graph sequences.
        if args.save_graphs:
            graph_sequence = []
            result.setdefault("graph_sequences", []).append(graph_sequence)
        else:
            graph_sequence = None

        # Initialize summary sequences.
        summaries = {}

        for step in range(args.num_lags):
            if args.save_graphs:
                graph_sequence.append(graph1.copy())
            collectiontools.append_values(
                summaries, simulator.evaluate_summaries(graph0, graph1)
            )
            # Skip simulation for the last step.
            if step == args.num_lags - 1:
                continue
            simulator.step(graph1)

        # Add summaries and parameters to the sequence.
        collectiontools.append_values(result.setdefault("summaries", {}), summaries)
        collectiontools.append_values(result.setdefault("params", {}), params)

    result["params"] = to_np_dict(result["params"])
    result["summaries"] = to_np_dict(result["summaries"])
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()

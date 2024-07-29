import argparse
import collectiontools
import numbers
from pathlib import Path
import pickle
from scipy import stats
from ..simulator import UniversalSimulator
from ..util import to_np_dict
from tqdm import tqdm


prior_clss = {"beta": stats.beta}

# See the `priors.ipynb` notebook for a more principled approach to choosing priors.
# Here, we simply use beta(2, 2) priors to push parameters away from the boundary.
prior_presets = {
    # Discrete-time stochastic simulator based on the continuous-time stochastic
    # simulator of Hansson et al. (2019) and the continous-time deterministic simulator
    # of Xiridou et al. (2003) obtained by setting :math:`\xi = 1`, i.e., serial
    # monogamy.
    "hansson2019": {
        "mu": stats.beta(2, 2),
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "w0": stats.beta(2, 2),
        "w1": stats.beta(2, 2),
        "xi": 0,
        "n": 200,
    },
    # Discrete-time stochastic simulator based on the stochastic discrete-time simulator
    # of Kretzschmar et al. (1996) obtained by setting :math:`\mu = 0` and
    # :math:`w_0 = w_1 = 0`, i.e., a closed population without casual contacts.
    "kretzschmar1996": {
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "xi": stats.beta(2, 2),
        "w0": 0,
        "w1": 0,
        "mu": 0,
        "n": 200,
    },
    # Discrete-time stochastic simulator based on the deterministic continuous-time
    # simulator of Leng et al. (2018) obtained by setting :math:`\mu = 0` and
    # :math:`\xi = 1`, i.e., a closed population and serial monogamy.
    "leng2018": {
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "xi": 0,
        "w0": stats.beta(2, 2),
        "w1": stats.beta(2, 2),
        "mu": 0,
        "n": 200,
    },
    # Discrete-time stochastic simulator based on Kretzschmar et al. (1998) obtained by
    # setting :math:`w_0 = w_1 = 0`, i.e., no casual sexual partners. They use a
    # different parameterization of the immigration rate :math:\nu = n / \mu` instead of
    # expected population size like we do here.
    "kretzschmar1998": {
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "xi": stats.beta(2, 2),
        "w0": 0,
        "w1": 0,
        "mu": stats.beta(2, 2),
        "n": 200,
    },
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


def __main__(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--burnin",
        help="number of burn in steps (defaults to `10 * n`)",
        type=int,
    )
    parser.add_argument(
        "--save_graphs",
        action="store_true",
        help="store graph sequences in addition to summaries (memory intensive)",
    )
    parser.add_argument(
        "--param",
        "-p",
        help="parameter value as [name]=[value] or prior as [name]=[prior_cls]:[*args]",
        action="append",
    )
    parser.add_argument("--preset", help="prior preset to use", choices=prior_presets)
    parser.add_argument("num_samples", help="number of samples", type=int)
    parser.add_argument("num_lags", help="number of lags to consider", type=int)
    parser.add_argument("output", help="output path for the samples", type=Path)
    args: Args = parser.parse_args(argv)

    priors = prior_presets[args.preset].copy()
    priors.update(parse_priors(args.param))
    extra = set(priors) - set(UniversalSimulator.arg_constraints)
    assert not extra, (
        f"Parameters {extra} are not allowed; allowed parameters are "
        f"{UniversalSimulator.arg_constraints}."
    )

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
        burnin = args.burnin or int(10 * params["n"])
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

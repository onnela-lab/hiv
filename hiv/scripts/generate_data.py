import argparse
import numpy as np
import pickle
from scipy import stats
from .. import stockholm
from tqdm import tqdm
import typing


def __main__(args: typing.Optional[typing.Iterable[str]] = None) -> None:
    # See the `priors.ipynb` notebook for a more principled approach to choosing priors. Here, we
    # simply use beta(2, 2) priors to push parameters away from the boundary.
    default_prior_args = {
        "mu": (2, 2),
        "sigma": (2, 2),
        "rho": (2, 2),
        "w0": (2, 2),
        "w1": (2, 2),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("num_samples", help="number of samples", type=int)
    parser.add_argument("n", help="expected number of nodes", type=int)
    parser.add_argument("num_lags", help="number of lags to consider", type=int)
    parser.add_argument("output", help="output path for the samples")
    parser.add_argument("--burnin", help="number of burn in steps (defaults to `10 * n`)", type=int)
    for param, prior_args in default_prior_args.items():
        parser.add_argument(f"--{param}_prior", type=lambda x: [float(x) for x in x.split(",")],
                            help=f"concentration parameters `(a, b)` for the beta prior of {param}",
                            default=prior_args)
        parser.add_argument(f"--{param}", type=float, help=f"value of {param}")
    args: argparse.Namespace = parser.parse_args(args)

    # Construct the priors and normalize parameters.
    priors = {param: stats.beta(*getattr(args, f"{param}_prior")) for param in default_prior_args}
    burnin = args.burnin or 10 * args.n

    graph_sequences = []
    param_sequence = {}
    for _ in tqdm(range(args.num_samples)):
        # Sample and use fixed values if provided.
        params = {key: prior.rvs() for key, prior in priors.items()} \
            | {key: value for key in priors if (value := getattr(args, key)) is not None}
        for key, param in params.items():
            param_sequence.setdefault(key, []).append(param)
        sequence = []
        graph = None
        for step in range(burnin + args.num_lags):
            graph, _ = stockholm.simulate(n=args.n, num_steps=1, step=step, graph=graph, **params)
            if step >= burnin:
                sequence.append(graph.copy())
        graph_sequences.append(sequence)

    with open(args.output, "wb") as fp:
        pickle.dump({
            "args": vars(args),
            "graph_sequences": graph_sequences,
            "param_sequence": {key: np.asarray(value) for key, value in param_sequence.items()},
        }, fp)


if __name__ == "__main__":
    __main__()

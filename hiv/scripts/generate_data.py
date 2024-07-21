import argparse
import networkx as nx
import pickle
from scipy import stats
from .. import stockholm
from ..util import to_np_dict
from tqdm import tqdm
import typing


def dict_append(container: dict, values: dict) -> None:
    for key, value in values.items():
        container.setdefault(key, []).append(value)


def __main__(args: typing.Optional[typing.Iterable[str]] = None) -> None:
    # See the `priors.ipynb` notebook for a more principled approach to choosing priors.
    # Here, we simply use beta(2, 2) priors to push parameters away from the boundary.
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
    parser.add_argument(
        "--burnin", help="number of burn in steps (defaults to `10 * n`)", type=int
    )
    parser.add_argument(
        "--save_graphs",
        action="store_true",
        help="store graph sequences in addition to summaries (memory intensive)",
    )
    for param, prior_args in default_prior_args.items():
        parser.add_argument(
            f"--{param}_prior",
            type=lambda x: [float(x) for x in x.split(",")],
            help=f"concentration parameters `(a, b)` for the beta prior of {param}",
            default=prior_args,
        )
        parser.add_argument(f"--{param}", type=float, help=f"value of {param}")
    args: argparse.Namespace = parser.parse_args(args)

    # Construct the priors and normalize parameters.
    priors = {
        param: stats.beta(*getattr(args, f"{param}_prior"))
        for param in default_prior_args
    }
    burnin = args.burnin or 10 * args.n

    result = {"args": vars(args)}
    for _ in tqdm(range(args.num_samples)):
        # Sample and use fixed values if provided.
        params = {key: prior.rvs() for key, prior in priors.items()} | {
            key: value for key in priors if (value := getattr(args, key)) is not None
        }
        for key, param in params.items():
            result.setdefault("params", {}).setdefault(key, []).append(param)

        # Run the burnin to get the first sample.
        graph0 = nx.empty_graph()
        for _ in range(burnin):
            graph0 = stockholm.step(graph0, n=args.n, **params)
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
            dict_append(summaries, stockholm.evaluate_summaries(graph0, graph1))
            # Skip simulation for the last step.
            if step == args.num_lags - 1:
                continue
            stockholm.step(graph1, n=args.n, **params)

        # Add summaries to the sequence.
        for key, values in summaries.items():
            result.setdefault("summaries", {}).setdefault(key, []).append(values)

    result["params"] = to_np_dict(result["params"])
    result["summaries"] = to_np_dict(result["summaries"])
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()

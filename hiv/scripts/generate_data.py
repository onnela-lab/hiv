import argparse
import collectiontools
import numbers
from pathlib import Path
import pickle
from scipy import stats
from ..simulators import KretzschmarMorris, Simulator, Stockholm
from ..util import to_np_dict
from tqdm import tqdm
from typing import Type


prior_clss = {"beta": stats.beta}
simulators_clss: dict[str, Type[Simulator]] = {
    "stockholm": Stockholm,
    "km": KretzschmarMorris,
}
# See the `priors.ipynb` notebook for a more principled approach to choosing priors.
# Here, we simply use beta(2, 2) priors to push parameters away from the boundary.
default_priors = {
    "stockholm": {
        "mu": stats.beta(2, 2),
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "w0": stats.beta(2, 2),
        "w1": stats.beta(2, 2),
        "n": 200,
    },
    "km": {
        "sigma": stats.beta(2, 2),
        "rho": stats.beta(2, 2),
        "xi": stats.beta(2, 2),
        "n": 200,
    },
}


class Args:
    simulator: str
    num_samples: int
    num_lags: int
    output: Path
    burnin: int
    store_graphs: bool
    param: list[str]
    save_graphs: bool


def __main__(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--burnin",
        help="number of burn in steps (defaults to `simulator.timescale`)",
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
    parser.add_argument("simulator", help="simulator to use", choices=simulators_clss)
    parser.add_argument("num_samples", help="number of samples", type=int)
    parser.add_argument("num_lags", help="number of lags to consider", type=int)
    parser.add_argument("output", help="output path for the samples", type=Path)
    args: Args = parser.parse_args(argv)

    simulator_cls = simulators_clss[args.simulator]
    priors = default_priors[args.simulator].copy()

    for param in args.param or []:
        parts = param.split("=")
        assert (
            len(parts) == 2
        ), f"Parameter specification `{param}` does not have format `[name]=[spec]`."
        arg, spec = parts
        assert arg in simulator_cls.arg_constraints, (
            f"Parameter `{arg}` is not allowed for {args.simulator}. Must be one of "
            f"{', '.join(simulator_cls.arg_constraints)}."
        )

        if ":" in spec:
            cls_name, hyperparams = spec.split(":")
            priors[arg] = prior_clss[cls_name](*map(float, hyperparams.split(",")))
        elif "." in spec:
            priors[arg] = float(spec)
        else:
            priors[arg] = int(spec)

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
        simulator = simulator_cls(**params)

        # Run the burnin to get the first sample.
        graph0 = simulator.init()
        burnin = 100
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

        # Add summaries to the sequence.
        for key, values in summaries.items():
            result.setdefault("summaries", {}).setdefault(key, []).append(values)
        collectiontools.append_values(result.setdefault("params", {}), params)

    result["params"] = to_np_dict(result["params"])
    result["summaries"] = to_np_dict(result["summaries"])
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()

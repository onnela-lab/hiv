import argparse
import collectiontools
import numpy as np
from pathlib import Path
import pickle
from ..simulator import DAYS_PER_YEAR


def __main__(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Output directory.")
    args = parser.parse_args(argv)

    # Create dataset in the same structure as what `generate_data` would generate.
    # Commented lines are present in the synthetic data but are omitted here.
    result = {
        # "args": vars(args),
        # "priors": priors,
        # "start": datetime.now(),
        # "end": end,
        # "duration": (end - result["start"]).total_seconds(),
    }

    # We create dummy parameters because the `run_abc` script expects them.
    params = ["n", "mu", "rho", "sigma", "omega0", "omega1", "xi"]
    result["params"] = {param: np.nan * np.ones(1) for param in params}

    # Manually add the summary statistics. For durations, we normalize by the length of
    # the year as in the simulations.
    summaries = {
        # From first paragraph in second column of page 70.
        "frac_paired": 0.64,
        # From S2.3 on page 3 of the supplement.
        "frac_concurrent": 0.146,
        # From the last paragraph in first column of page 70.
        "steady_length": 203.2 / DAYS_PER_YEAR,
        # From about half-way down the second column of page 70 just after the equation.
        "casual_gap_paired": 101.9 / DAYS_PER_YEAR,
        "casual_gap_single": 62.6 / DAYS_PER_YEAR,
        # Not part of the Hansson et al. dataset.
        "frac_retained_nodes": np.nan,
        "frac_single_with_casual": np.nan,
        "frac_retained_steady_edges": np.nan,
        "frac_paired_with_casual": np.nan,
    }
    # We need to get the features to have shape `(batch_size, n_lags)`.
    result["summaries"] = collectiontools.map_values(np.atleast_2d, summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()

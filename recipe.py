from cook import create_task, Task
import hashlib
import itertools
import os
from pathlib import Path
from typing import Any


CI = "CI" in os.environ
workspace = Path("workspace")


def create_generate_data_task(
    *,
    seed: int,
    burnin: int,
    debug: bool,
    target: Path,
    task_name: str,
    batch_size: int,
    num_lags: int,
    preset: str,
    priors: dict[str, Any],
) -> Task:
    action = [
        "python",
        "-m",
        "hiv.scripts.generate_data",
        f"--seed={seed}",
        f"--preset={preset}",
        f"--burnin={burnin}",
        batch_size,
        num_lags,
        target,
    ]
    for arg, spec in priors.items():
        action.append(f"--param={arg}={spec}")

    if debug:
        action.append("--save-graphs")
        action = [
            "python",
            "-m",
            "cProfile",
            "-o",
            target.with_suffix(".prof"),
        ] + action[1:]
    return create_task(task_name, action=action, targets=[target])


def str2seed(value: str | Path) -> int:
    """
    Get a 32-bit seed based on the hash of the target. This will ensure distinct seeds
    across all runs (with high probability).
    """
    seed_bytes = hashlib.sha256(str(value).encode()).digest()
    return int.from_bytes(seed_bytes[:4], "little")


def main():
    create_task(
        "tests",
        action="pytest -v --cov=hiv --cov-report=term-missing --cov-fail-under=100",
    )
    create_task("lint", action="black --check .")

    # Create the data from Hansson et al. in the right format.
    hansson2019data = workspace / "empirical" / "hansson2019data.pkl"
    create_task(
        "empirical/hansson2019data",
        action=["python", "-m", "hiv.scripts.create_hansson_data", hansson2019data],
        targets=[hansson2019data],
        dependencies=["src/hiv/scripts/create_hansson_data.py"],
    )

    # Tuple of (num_batches, batch_size) for more efficient generation of training data.
    split_sizes = {
        "debug": (1, 10),
        "test": (10, 100),
        "validation": (1, 100),
        "train": (100, 1000),
    }
    configs = {
        "default": {},
        "kretzschmar1996": {},
        "kretzschmar1998": {},
        "leng2018": {},
        "hansson2019": {},
        "medium": {
            "mu": "beta:2,2",
            "rho": "beta:2,2",
            "sigma": "beta:2,2",
            "omega0": "beta:2,2",
            "omega1": "beta:2,2",
            "xi": "0",
            "n": "200",
        },
        "small": {
            "mu": "beta:2,18",
            "rho": "beta:2,18",
            "sigma": "beta:2,18",
            "omega0": "beta:2,18",
            "omega1": "beta:2,18",
            "xi": "0",
            "n": "200",
        },
        "x-small": {
            "mu": "beta:2,48",
            "rho": "beta:2,48",
            "sigma": "beta:2,48",
            "omega0": "beta:2,48",
            "omega1": "beta:2,48",
            "xi": "0",
            "n": "200",
        },
    }
    small_burnin_presets = {"medium", "small", "x-small"}
    # Models run at *weekly* scales. We consider up to five year lags.
    num_lags = 5 if CI else 5 * 52

    # Iterate over different models.
    for preset, priors in configs.items():
        burnin = 5 * 52 if preset in small_burnin_presets else 52 * 30
        # Iterate over different splits.
        batches_by_split: dict[str, list] = {}
        for split, (num_batches, batch_size) in split_sizes.items():
            for batch in range(num_batches):
                task_name = f"{preset}/{split}/{batch}"
                target = (workspace / task_name).with_suffix(".pkl")
                create_generate_data_task(
                    seed=str2seed(target),
                    burnin=burnin,
                    batch_size=batch_size,
                    num_lags=num_lags,
                    target=target,
                    task_name=task_name,
                    debug=split == "debug",
                    preset="default" if priors else preset,
                    priors=priors,
                )
                batches_by_split.setdefault(split, []).append(target)

        # Run inference on the batches with different configurations.
        for adjust, standardize, summaries in itertools.product(
            [False, True],
            [None, "global", "local"],
            ["all", "hansson-only", "frac-only"],
        ):
            parts = [
                "adjusted" if adjust else "unadjusted",
                standardize if standardize else "none",
                summaries,
            ]
            task_name = "/".join([preset, "inference"] + parts)

            argv: list[str | Path] = ["python", "-m", "hiv.scripts.run_abc"]
            if adjust:
                argv.append("--adjust")
            if standardize:
                argv.append(f"--standardize={standardize}")

            # The summaries we're going to exclude for different configurations, e.g.,
            # all the temporal features if we're limiting to fractions only.
            exclude = {
                "all": [],
                "hansson-only": [
                    "frac_retained_steady_edges",
                    "frac_retained_nodes",
                    "frac_single_with_casual",
                    "frac_paired_with_casual",
                ],
                "frac-only": [
                    "steady_length",
                    "casual_gap_single",
                    "casual_gap_paired",
                ],
            }
            argv.extend(f"--exclude={summary}" for summary in exclude[summaries])

            target = (workspace / preset / "-".join(["result"] + parts)).with_suffix(
                ".pkl"
            )
            argv.extend(
                [workspace / preset / "train", workspace / preset / "test", target]
            )
            create_task(
                task_name,
                action=argv,
                targets=[target],
                dependencies=batches_by_split["train"] + batches_by_split["test"],
            )

        # Run configurations with different population sizes for a fixed sample size to
        # diagnose sensitivity of statistics to population size. We use the same sample size
        # as the data from Hansson et al. (2019) for their study of MSM in Stockholm.
        sample_size = 403
        population_sizes = [sample_size * i for i in [1, 2, 3, 5, 10, 20]]
        for population_size in population_sizes:
            task_name = f"{preset}/size_sensitivity/size_{population_size}"
            target = (workspace / task_name).with_suffix(".pkl")
            create_generate_data_task(
                # We use the same seed for all of these to try and get at variability
                # due to just the population size, not random variation.
                seed=str2seed(target.parent),
                burnin=60 * 52,
                batch_size=batch_size,
                num_lags=num_lags,
                target=target,
                task_name=task_name,
                debug=False,
                preset=preset,
                priors=priors | {"n": population_size},
            )

        # Run different presets against the empirical data.
        task_name = f"empirical/inference/{preset}"
        target = workspace / f"empirical/samples/{preset}.pkl"
        argv: list[str | Path] = [
            "python",
            "-m",
            "hiv.scripts.run_abc",
            "--adjust",
            "--save-samples",
            "--exclude=frac_retained_steady_edges",
            "--exclude=frac_retained_nodes",
            "--exclude=frac_single_with_casual",
            "--exclude=frac_paired_with_casual",
            workspace / preset / "train",
            hansson2019data.parent,
            target,
        ]
        create_task(
            name=task_name,
            targets=[target],
            dependencies=[*batches_by_split["train"], hansson2019data],
            action=argv,
        )


main()

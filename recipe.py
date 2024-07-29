from cook import create_task
import os
from pathlib import Path


CI = "CI" in os.environ

workspace = Path("workspace")


create_task(
    "tests", action="pytest -v --cov=hiv --cov-report=term-missing --cov-fail-under=100"
)
create_task("lint", action="black --check .")

split_sizes = {
    "debug": 10,
    "test": 1_000,
    "train": 10_000,
}
configs = {
    "hansson2019": {
        "medium": {
            "n": 200,
            "mu": "beta:2,2",
            "rho": "beta:2,2",
            "sigma": "beta:2,2",
            "w0": "beta:2,2",
            "w1": "beta:2,2",
        },
        "small": {
            "n": 200,
            "mu": "beta:2,18",
            "rho": "beta:2,18",
            "sigma": "beta:2,18",
            "w0": "beta:2,18",
            "w1": "beta:2,18",
        },
        "x-small": {
            "n": 200,
            "mu": "beta:2,48",
            "rho": "beta:2,48",
            "sigma": "beta:2,48",
            "w0": "beta:2,48",
            "w1": "beta:2,48",
        },
    },
    "kretzschmar1998": {
        "medium": {
            "n": 200,
            "rho": "beta:2,2",
            "sigma": "beta:2,2",
            "xi": "beta:2,2",
        },
        "small": {
            "n": 200,
            "rho": "beta:2,18",
            "sigma": "beta:2,18",
            "xi": "beta:2,18",
        },
        "x-small": {
            "n": 200,
            "rho": "beta:2,48",
            "sigma": "beta:2,48",
            "xi": "beta:2,48",
        },
    },
}
# Models run at weekly scales. We consider up to five year lags.
num_lags = 5 if CI else 5 * 52

# Iterate over different models.
for preset, priors in configs.items():
    # Iterate over different prior configurations for this model.
    for name, prior in priors.items():
        # Iterate over different splits.
        for split, size in split_sizes.items():
            task_name = f"{preset}/{name}/{split}"
            target = (workspace / task_name).with_suffix(".pkl")
            action = [
                "python",
                "-m",
                "hiv.scripts.generate_data",
                f"--preset={preset}",
                size,
                num_lags,
                target,
            ]
            for arg, spec in prior.items():
                action.append(f"--param={arg}={spec}")

            if split == "debug":
                action.append("--save_graphs")
                action = [
                    "python",
                    "-m",
                    "cProfile",
                    "-o",
                    target.with_suffix(".prof"),
                ] + action[1:]
            create_task(task_name, action=action, targets=[target])

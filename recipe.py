from cook import create_task
import hashlib
import os
from pathlib import Path


CI = "CI" in os.environ

workspace = Path("workspace")


create_task(
    "tests", action="pytest -v --cov=hiv --cov-report=term-missing --cov-fail-under=100"
)
create_task("lint", action="black --check .")

# Tuple of (num_batches, batch_size) for more efficient generation of training data.
split_sizes = {
    "debug": (1, 10),
    "test": (10, 100),
    "validation": (1, 100),
    "train": (10, 1000),
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
    },
    "small": {
        "mu": "beta:2,18",
        "rho": "beta:2,18",
        "sigma": "beta:2,18",
        "omega0": "beta:2,18",
        "omega1": "beta:2,18",
    },
    "x-small": {
        "mu": "beta:2,48",
        "rho": "beta:2,48",
        "sigma": "beta:2,48",
        "omega0": "beta:2,48",
        "omega1": "beta:2,48",
    },
}
# Models run at weekly scales. We consider up to five year lags.
num_lags = 5 if CI else 5 * 52

# Iterate over different models.
for preset, prior in configs.items():
    # Iterate over different splits.
    for split, (num_batches, batch_size) in split_sizes.items():
        for batch in range(num_batches):
            task_name = f"{preset}/{split}/{batch}"
            target = (workspace / task_name).with_suffix(".pkl")
            # Get a 32-bit seed based on the hash of the target. This will ensure
            # distinct seeds across all runs (with high probability).
            seed_bytes = hashlib.sha256(str(target).encode()).digest()
            seed = int.from_bytes(seed_bytes[:4], "little")
            action = [
                "python",
                "-m",
                "hiv.scripts.generate_data",
                f"--seed={seed}",
                f"--preset={'default' if prior else preset}",
                # Burn in for thirty years.
                f"--burnin={52 * 30}",
                batch_size,
                num_lags,
                target,
            ]
            for arg, spec in prior.items():
                action.append(f"--param={arg}={spec}")

            if split == "debug":
                action.append("--save-graphs")
                action = (
                    [
                        "python",
                        "-m",
                        "cProfile",
                        "-o",
                        target.with_suffix(".prof"),
                    ]
                    + action[1:]
                )
            create_task(task_name, action=action, targets=[target])

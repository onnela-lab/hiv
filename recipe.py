from cook import create_task
import hashlib
import itertools
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
for preset, prior in configs.items():
    # Iterate over different splits.
    batches_by_split: dict[str, list] = {}
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
                f"--burnin={5 * 52 if preset in small_burnin_presets else 52 * 30}",
                batch_size,
                num_lags,
                target,
            ]
            for arg, spec in prior.items():
                action.append(f"--param={arg}={spec}")

            if split == "debug":
                action.append("--save-graphs")
                action = [
                    "python",
                    "-m",
                    "cProfile",
                    "-o",
                    target.with_suffix(".prof"),
                ] + action[1:]
            create_task(task_name, action=action, targets=[target])
            batches_by_split.setdefault(split, []).append(target)

    # Run inference on the batches with different configurations.
    for adjust, standardize in itertools.product(
        [False, True], [None, "global", "local"]
    ):
        parts = [
            "adjusted" if adjust else "unadjusted",
            standardize if standardize else "none",
        ]
        task_name = "/".join([preset, "inference"] + parts)

        argv = ["python", "-m", "hiv.scripts.run_abc"]
        if adjust:
            argv.append("--adjust")
        if standardize:
            argv.append(f"--standardize={standardize}")

        target = (workspace / preset / "-".join(["result"] + parts)).with_suffix(".pkl")
        argv.extend([workspace / preset / "train", workspace / preset / "test", target])
        create_task(
            task_name,
            action=argv,
            targets=[target],
            dependencies=batches_by_split["train"] + batches_by_split["test"],
        )

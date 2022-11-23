import doit_interface as di


manager = di.Manager.get_instance()

# Manage python requirements.
manager(basename="requirements", name="txt", targets=["requirements.txt"],
        file_dep=["requirements.in"], actions=["pip-compile"])
manager(basename="requirements", name="sync", file_dep=["requirements.txt"], actions=["pip-sync"])

# Generate different sample sizes with optional custom parameters.
split_sizes = {
    "debug": (100, None),
    "test": (1_000, None),
    "train": (10_000, None),
    "debug-fixed": (100, {"mu": 0.1, "sigma": 0.2, "rho": 0.4, "w0": 0.2, "w1": 0.1}),
}
n = 200

# Different prior parameterizations. The smaller the parameters, the longer the time scale over
# which the graphs evolve.
priors = {
    "medium": {
        "mu": (2, 2),
        "rho": (2, 2),
        "sigma": (2, 2),
        "w0": (2, 2),
        "w1": (2, 2),
    },
    "small": {
        "mu": (2, 18),
        "rho": (2, 18),
        "sigma": (2, 18),
        "w0": (2, 18),
        "w1": (2, 18),
    },
    "x-small": {
        "mu": (2, 48),
        "rho": (2, 48),
        "sigma": (2, 48),
        "w0": (2, 48),
        "w1": (2, 48),
    },
}

for name, prior in priors.items():
    for split, (size, kwargs) in split_sizes.items():
        target = f"workspace/{name}/{split}.pkl"
        action = ["python", "-m", "hiv.scripts.generate_data", size, n, 100, target]
        if split.startswith("debug"):
            action.append("--save_graphs")
        kwargs = kwargs or {}
        kwargs.update({f"{key}_prior": ",".join(map(str, params)) for key, params in prior.items()})
        action.extend(di.dict2args(kwargs))
        manager(basename=f"simulations-{name}", name=split, targets=[target], actions=[action],
                uptodate=[True])

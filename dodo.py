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

for split, (size, kwargs) in split_sizes.items():
    target = f"workspace/{split}.pkl"
    action = ["python", "-m" "hiv.scripts.generate_data", size, n, 50, target]
    action.extend(di.dict2args(kwargs or {}))
    manager(basename="simulations", name=split, targets=[target], actions=[action])

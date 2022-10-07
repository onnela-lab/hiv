import doit_interface as di


manager = di.Manager.get_instance()

# Manage python requirements.
manager(basename="requirements", name="txt", targets=["requirements.txt"],
        file_dep=["requirements.in"], actions=["pip-compile"])
manager(basename="requirements", name="sync", file_dep=["requirements.txt"], actions=["pip-sync"])

# Generate different sample sizes.
split_sizes = {
    "debug": 100,
    "test": 1_000,
    "train": 10_000,
}
n = 200

for split, size in split_sizes.items():
    target = f"workspace/{split}.pkl"
    manager(basename="simulations", name=split, targets=[target],
            actions=[["python", "generate_data.py", size, n, 50, target]])

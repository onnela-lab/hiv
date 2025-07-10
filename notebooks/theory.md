---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

This notebook verifies that simulations agree with theoretical results where it is possible to derive them. We largely focus on the setup of Hansson et al. (2019) who use the full simulations with the exception of concurrency not being part of the model.

```{code-cell} ipython3
import collectiontools
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

from hiv.simulator import estimate_paired_fraction
```

```{code-cell} ipython3
# Load all the chunks and concatenate them into parameters and summaries.
filenames = list(sorted(Path("../workspace/hansson2019/test").glob("*.pkl")))
params = {}
summaries = {}
for filename in tqdm(filenames):
    chunk = pd.read_pickle(filename)
    collectiontools.append_values(params, chunk["params"])
    collectiontools.append_values(
        summaries,
        {
            key: value for key, value in chunk["summaries"].items()
            if not key.startswith("_") or key == "_frac_paired"
        },
    )

params = collectiontools.map_values(np.concatenate, params)
summaries = collectiontools.map_values(np.concatenate, summaries)

lines = [
    f"Loaded {len(params['rho'])} instances from {len(filenames)} files.",
    f"Parameters: {', '.join(params)}.",
    f"Summaries: {', '.join(summaries)}.",
]
print("\n".join(lines))
```

```{code-cell} ipython3
# Show plots of simulated statistics against expected values. The expected values are a
# mapping {key: [(expected_value, lag), ...]}. key may either be a summary name or a
# tuple of summary name and an index (which is applied at the end like
# [..., lag, index]). This is useful for extracting debug statistics like the fraction
# of paired nodes at a particular lag (as opposed to averaged over the two waves).

# For the steady length, we need to sample. The probability for the relationship to
# survive is:
proba = (1 - params["sigma"]) * (1 - params["mu"]) ** 2
# We can sample the time until failure, i.e., how long the relationship survives.
x = np.random.geometric(1 - proba, size=(2001, *proba.shape))
x = x.clip(max=52).mean(axis=0) / 52

expected_values_by_lag = {
    ("_frac_paired", 1): [
        (estimate_paired_fraction(params["rho"], params["mu"], params["sigma"]), 0),
        (estimate_paired_fraction(params["rho"], params["mu"], params["sigma"], lag=4 * 52), 4 * 52),
    ],
    "frac_paired_with_casual": [(params["omega1"], 0), (params["omega1"], 52)],
    "frac_single_with_casual": [(params["omega0"], 0), (params["omega0"], 52)],
    "steady_length": [(x, 0)],
}

fig, axes = plt.subplots(2, 2)
for ax, (key, expected_values) in zip(axes.ravel(), expected_values_by_lag.items()):
    lower = np.inf
    upper = - np.inf
    for expected, lag in expected_values:
        if isinstance(key, tuple):
            name, idx = key
            actual = summaries[name][..., lag, idx]
        else:
            name = key
            actual = summaries[key][..., lag]
        ax.scatter(expected, actual, marker=".", label=f"$\\tau={lag}$")
        lower = min(lower, expected.min())
        upper = max(upper, expected.max())

        # Report the correlation coefficient for the values.
        corrcoef = np.corrcoef(expected, actual)[0, 1]
        print(f"Correlation coefficient for {key} at lag {lag}: {corrcoef}.")


    # Diagonal line and setup.
    ax.plot((lower, upper), (lower, upper), color="k", ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("expected")
    ax.set_ylabel("simulated")
    ax.set_title(name)
    ax.legend(fontsize="x-small")
fig.tight_layout()
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2)

# From Hansson et al. (2019).
refs = {
    "casual_gap_single": 23.1,
    "casual_gap_paired": 36.3,
}

for ax, (param, summary) in zip(axes, [("omega0", "casual_gap_single"), ("omega1", "casual_gap_paired")]):
    ax.scatter(params[param], summaries[summary][:, 0])
    ax.axhline(refs[summary] / 365.25)
    ax.set_xlabel(param)
    ax.set_ylabel(summary)

fig.tight_layout()
```

```{code-cell} ipython3
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, :])

x = np.arange(summaries["frac_paired"].shape[1])

ys1 = summaries["_frac_paired"][..., 1]
loc = ys1.mean(axis=0)
scale = 1.96 * ys1.std(axis=0) / np.sqrt(ys1.shape[0])
line, = ax.plot(x, loc, label="simulation")
ax.fill_between(x, loc - scale, loc + scale, color=line.get_color(), alpha=0.2)

ys2 = estimate_paired_fraction(params["rho"], params["mu"], params["sigma"], lag=x[:, None]).T
loc = ys2.mean(axis=0)
scale = 1.96 * ys2.std(axis=0) / np.sqrt(ys2.shape[0])
line, = ax.plot(x, loc, label="theory")
ax.fill_between(x, loc - scale, loc + scale, color=line.get_color(), alpha=0.2)

ax.set_xlabel("lag $\\tau$")
ax.set_ylabel("fraction of paired nodes")
ax.legend()

indices = (params["mu"].argmin(), params["mu"].argmax())
for i, idx in enumerate(indices):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(x, ys1[idx], label="simulation")
    ax.plot(x, ys2[idx], label="theory")

    ax.set_xlabel("lag $\\tau$")
    ax.set_ylabel("fraction of paired nodes")
    ax.set_title(f"mu = {params['mu'][idx]:.2g}")
    ax.legend()

fig.tight_layout()
```

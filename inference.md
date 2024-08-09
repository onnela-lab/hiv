---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import collectiontools
from pathlib import Path
from scipy import stats
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from typing import Hashable
from hiv.scripts.generate_data import parse_priors
import itertools as it
```

```{code-cell} ipython3
def flatten_dict(x: dict[Hashable, np.ndarray], moveaxis=True) -> np.ndarray:
    """
    Flatten a dictionary, ensuring consistent key ordering.
    """
    x = np.asarray([value for _, value in sorted(x.items())])
    if moveaxis:
        x = np.moveaxis(x, 0, -1)
    return x
```

```{code-cell} ipython3
SPLITS = ["train", "test"]
PRESETS = ["medium", "small", "x-small"]


def load(preset: str, exclude_params=None, exclude_summaries=None) -> tuple[dict, dict]:
    """
    Load summaries and parameters keyed by different splits.
    """
    summaries_by_split = {}
    params_by_split = {}
    exclude_params = exclude_params or set()
    exclude_summaries = exclude_summaries or set()

    for split in SPLITS:
        filenames = list((Path("workspace") / preset / split).glob("*.pkl"))
        assert filenames, (model, prior, split)
        for filename in filenames:
            with filename.open("rb") as fp:
                result = pickle.load(fp)
                
            summaries = {
                key: value for key, value in result["summaries"].items() 
                if not (key.startswith("_") or key in exclude_summaries)
            }
            collectiontools.append_values(
                summaries_by_split.setdefault(split, {}), 
                summaries,
            )
            
            params = {
                key: value for key, value in result["params"].items() 
                if not (key == "n" or key in exclude_params)
            }
            collectiontools.append_values(
                params_by_split.setdefault(split, {}), 
                params,
            )
        
    return {
        "summaries_by_split": {
            key: collectiontools.map_values(np.concatenate, value) 
            for key, value in summaries_by_split.items()
        },
        "params_by_split": {
            key: collectiontools.map_values(np.concatenate, value) 
            for key, value in params_by_split.items()
        },
        "priors": result["priors"],
        "args": result["args"],
    }


results_by_preset = {}
for preset in PRESETS:
    result = load(preset, exclude_params={"xi"}, exclude_summaries={"frac_concurrent"}) 
    results_by_preset[preset] = result

    print("; ".join([
        preset,
        f"train_summaries.shape = {flatten_dict(result['summaries_by_split']['train']).shape}",
    ]))
```

```{code-cell} ipython3
params_stats = [
    ("mu", "frac_retained_nodes"),
    ("sigma", "frac_retained_steady_edges"),
    ("rho", "frac_paired"),
    ("xi", "frac_concurrent"),
    ("omega0", "frac_single_with_casual"),
    ("omega1", "frac_paired_with_casual"),
]

preset = "x-small"
split = "test"
lag = 200

result = results_by_preset[preset]
params = result["params_by_split"][split]
stats = result["summaries_by_split"][split]

fig, axes = plt.subplots(2, 3)
for ax, (param, stat) in zip(axes.ravel(), sorted(params_stats)):
    ax.set_xlabel(param)
    ax.set_ylabel(stat)
    try:
        ax.scatter(
            params[param], 
            stats[stat][:, lag], marker="."
        )
    except KeyError:
        pass
    
fig.tight_layout()

params.keys(), stats.keys()
```

```{code-cell} ipython3
def evaluate_mses(
    result: dict,
    frac: float = 0.01,
    standardize: bool = False,
    aggregate: bool = True,
) -> np.ndarray:
    """
    Evaluate the mean squared error under the posterior distribution.

    Args:
        result: Result dictionary containing
            train_summaries: Summary statistics in the reference table.
            train_params: Parameters that generated summaries in the reference table.
            test_summaries: Summary statistics evaluated on the test set.
            test_params: Parameters that generated the test set.
        frac: Fraction of reference table to sample.
        standardize: Standardize features.
        aggregate: Aggregate the mean squared error over all parameters.

    Returns:
        Mean-squared errors with shape `(num_lags, num_test_samples)` if `aggregate` 
        else `(num_lags, num_test_samples, num_params)`.
    """
    train_summaries = flatten_dict(result["summaries_by_split"]["train"])
    train_params = flatten_dict(result["params_by_split"]["train"])
    test_summaries = flatten_dict(result["summaries_by_split"]["test"])
    test_params = flatten_dict(result["params_by_split"]["test"])
    
    # Get shape parameters.
    num_train_samples, num_lags, num_summaries = train_summaries.shape
    num_test_samples, _num_lags, num_summaries = test_summaries.shape
    num_abc = int(num_train_samples * frac)

    # Mean-subtract and scale to unit variance.
    if standardize:
        scaler = StandardScaler()
        shape = train_summaries.shape
        assert shape[0] == num_summaries, (shape, num_summaries)
        train_summaries = scaler.fit_transform(
            train_summaries.T.reshape((-1, _num_summaries))
        ).T.reshape(shape)

        shape = test_summaries.shape
        assert shape[0] == num_summaries, (shape, num_summaries)
        test_summaries = scaler.transform(
            test_summaries.T.reshape((-1, num_summaries))
        ).T.reshape(shape)

    mses = []
    for lag in range(num_lags):
        tree = KDTree(train_summaries[:, lag])
        distances, indices = tree.query(test_summaries[:, lag], k=num_abc)
        samples_abc = train_params[indices]
        residuals = samples_abc - test_params[:, None]
        # Residuals have shape (num_test_samples, num_posterior_samples, num_params).
        if aggregate:
            mse = np.square(residuals).mean(axis=(1, 2))
        else:
            mse = np.square(residuals).mean(axis=1)
        mses.append(mse)

    mses = np.asarray(mses)
    if aggregate:
        assert mses.shape == (num_lags, num_test_samples), mses.shape
    else:
        assert mses.shape == (num_lags, num_test_samples, test_params.shape[-1]), mses.shape
    return mses

# Precompute the mses.
mses_by_preset = {
    preset: evaluate_mses(result, standardize=False, aggregate=False) 
    for preset, result in results_by_preset.items()
}
```

```{code-cell} ipython3
preset = "x-small"
mses = mses_by_preset[preset]
assert mses.ndim == 3
labels = list(sorted(results_by_preset[preset]["params_by_split"]["test"]))

fig, axes = plt.subplots(2, 3, sharex=True)

for ax, ys, label in zip(axes.ravel(), mses.T, labels):
    # ys has shape (n_test, n_lags).
    y = ys.mean(axis=0)
    yerr = ys.std(axis=0) / np.sqrt(ys.shape[0] - 1)
    x = np.arange(y.size)
    line, = ax.plot(x, y)
    ax.fill_between(x, y - yerr, y + yerr, color=line.get_color(), alpha=0.2)
    ax.yaxis.major.formatter.set_powerlimits((0, 0))
    ax.set_ylabel(label)
    ax.axhline(y[0], color="k", ls=":")

fig.tight_layout()
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

factor = 1.96
show_prior_mse = False

for ax, (preset, result) in zip(axes[1:], results_by_preset.items()):
    
    # Show the prior distribution.
    dist = result["priors"]["mu"]
    x = np.linspace(0, 1, 200)
    line, = axes[0].plot(x, dist.pdf(x), label=f"Beta({int(dist.args[0])}, {int(dist.args[1])})")
    color = line.get_color()

    # Show the mean-squared errors as a function of lag.
    mses = mses_by_preset[preset]
    if mses.ndim == 3:
        mses = mses.mean(axis=-1)
    y = mses.mean(axis=1)
    yerr = factor * mses.std(axis=1) / np.sqrt(mses.shape[1] - 1)
    x = np.arange(y.size)
    line, = ax.plot(x, y, color=color)
    ax.fill_between(x, y - yerr, y + yerr, color=line.get_color(), alpha=.2)
    ax.axhline(y[0], color="k", ls=":")
    ax.yaxis.major.formatter.set_useMathText(True)
    ax.yaxis.major.formatter.set_powerlimits((0, 0))
    ax.set_xlabel("lag")
    ax.set_ylabel("rmse")

    # Show the prior mse.
    if show_prior_mse:
        c, d = np.split(params_by_split["test"], 2, axis=1)
        prior_mses = np.square(c, d).mean(axis=0)
        prior_y = np.mean(prior_mses)
        prior_yerr = factor * np.std(prior_mses) / np.sqrt(prior_mses.size - 1)
        line = ax.axhline(prior_y, color="gray", ls="--", label="prior")
        ax.axhspan(prior_y - prior_yerr, prior_y + prior_yerr,
                   color=line.get_color(), alpha=.2)

    # Add a "guess" of where we'd expect the best lag to be.
    proba = (1 - dist.mean()) ** x
    guess = np.argmin(np.abs(proba - 0.5))
    ax.axvline(guess, color="k", ls=":", zorder=0)

# Add labels and legends.
ax = axes[0]
ax.legend(loc="center right")
ax.set_ylabel(r"prior density $p(\theta)$")
ax.set_xlabel(r"parameters $\theta$")
text = ax.text(0.95, 0.95, "(a)", transform=ax.transAxes,
               ha="right", va="top")

for ax, label in zip(axes[1:], "bcd"):
    text = ax.text(0.95, 0.05, f"({label})", transform=ax.transAxes,
                   ha="right", va="bottom")
fig.tight_layout()
```

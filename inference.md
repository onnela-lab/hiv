---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
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
from scipy import stats
from scipy.spatial import KDTree
from typing import Hashable
from hiv.scripts.generate_data import parse_priors
```

```{code-cell} ipython3
def flatten_dict(x: dict[Hashable, np.ndarray]) -> np.ndarray:
    """
    Flatten a dictionary with consistent key ordering.
    """
    return np.asarray([value for _, value in sorted(x.items())])
```

```{code-cell} ipython3
SPLITS = ["train", "test"]
PRIORS = ["medium", "small", "x-small"]


def load(prior: str, model: str) -> tuple[dict, dict]:
    """
    Load summaries and parameters keyed by different splits.
    """
    summaries_by_split = {}
    params_by_split = {}

    for split in SPLITS:
        with open(f"workspace/{model}/{prior}/{split}.pkl", "rb") as fp:
            result = pickle.load(fp)
        summaries_by_split[split] = flatten_dict(result["summaries"])
        params_by_split[split] = flatten_dict(result["params"])

    return summaries_by_split, params_by_split, result["args"]


results_by_prior = {prior: load(prior, "stockholm") for prior in PRIORS}
```

```{code-cell} ipython3
def evaluate_mses(train_summaries: np.ndarray, train_params: np.ndarray, test_summaries: np.ndarray,
                  test_params: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """
    Evaluate the mean squared error under the posterior distribution.

    Args:
        train_summaries: Summary statistics in the reference table.
        train_params: Parameters that generated summaries in the reference table.
        test_summaries: Summary statistics evaluated on the test set.
        test_params: Parameters that generated the test set.
        frac: Fraction of reference table to sample.

    Returns:
        Mean-squared errors with shape `(num_lags, num_test_samples)`.
    """
    # Get shape parameters.
    _num_summaries, num_train_samples, num_lags = train_summaries.shape
    _num_summaries, num_test_samples, _num_lags = test_summaries.shape
    num_abc = int(num_train_samples * frac)

    mses = []
    for lag in range(num_lags):
        tree = KDTree(train_summaries[..., lag].T)
        distances, indices = tree.query(test_summaries[..., lag].T, k=num_abc)
        samples_abc = train_params[:, indices]
        residuals = samples_abc - test_params[..., None]
        mse = np.square(residuals).mean(axis=(0, 2))
        mses.append(mse)

    mses = np.asarray(mses)
    assert mses.shape == (num_lags, num_test_samples)
    return mses

# Precompute the mses.
mses_by_prior = {
    prior: evaluate_mses(
        summaries_by_split["train"],
        params_by_split["train"],
        summaries_by_split["test"],
        params_by_split["test"],
    ) for prior, (summaries_by_split, params_by_split, args)
    in results_by_prior.items()
}
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

factor = 1.96
show_prior_mse = False

for ax, (prior, (summaries_by_split, params_by_split, args)) \
        in zip(axes[1:], results_by_prior.items()):
    # Show the prior distribution.
    priors = parse_priors(args["param"])
    dist = priors["mu"]
    x = np.linspace(0, 1, 200)
    line, = axes[0].plot(x, dist.pdf(x), label=f"Beta({int(dist.args[0])}, {int(dist.args[1])})")
    color = line.get_color()

    # Show the mean-squared errors as a function of lag.
    mses = mses_by_prior[prior]
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

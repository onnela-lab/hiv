---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
from hiv.scripts import generate_data
from hiv.simulator import estimate_paired_fraction
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 144
```

```{code-cell} ipython3
# Here we use the reference values from Hansson and then vary one of the parameters
# using the prior from our main simulation. These are taken from the inference notebook.
reference = {
    "rho": 0.042036,
    "sigma": 0.023064,
    "mu": 0.000319,
}
priors = {
    "mu": stats.beta(2.7471745380604844, 2162.302624350188),
    "rho": stats.beta(1.898530545859645, 24.62575083539298),
    "sigma": stats.beta(2.0596220554513454, 98.67884889543724),
}

fig, axes = plt.subplots(2, 2)
for ax, (key, value) in zip(axes.ravel(), reference.items()):
    ax.axvline(value)
    ax.set_title(key)
    prior = priors[key]
    l, u = prior.ppf([0.001, 0.999])
    x = np.linspace(l, u, 500)
    ax.plot(x, prior.pdf(x))

fig.tight_layout()
```

```{code-cell} ipython3
# Run all the simulations we need for plotting.
force = False

def _create_args(params: dict, path, num_samples=500, sample_size=100):
    params = [f"--param={key}={value}" for key, value in params.items()]
    return [
        "--seed=42",
        "--preset=empty",
        f"--sample-size={sample_size}",  # Number of people in cohort.
        *params,
        str(num_samples),  # Number of samples.
        "260",  # Number of lag steps: 5 years.
        str(path),
    ]

default_params = {
    "n": 500,
    "xi": 0,
    "omega0": 0,
    "omega1": 0,
}
simulations = {}
parent = Path("../workspace/thermodynamic-limit")
parent.mkdir(exist_ok=True, parents=True)
for key, value in reference.items():
    path = parent / f"{key}.pkl"
    if not path.is_file() or force:
        params = default_params.copy()
        for other, prior in priors.items():
            if key == other:
                params[other] = f"beta:{prior.args[0]},{prior.args[1]}"
            else:
                params[other] = reference[other]
        args = _create_args(params, path)
        print(path, args)
        generate_data.__main__(args)
    simulations[key] = pd.read_pickle(path)

for n in [500, 1000, 5000]:
    path = parent / f"all-fixed-{n}.pkl"
    all_fixed = reference | {"mu": 0.01}
    if not path.is_file() or force:
        params = default_params | all_fixed | {"n": n}
        print(path)
        generate_data.__main__(_create_args(params, path, num_samples=1000))
    simulations[path.stem] = pd.read_pickle(path)
simulations.keys()
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2)

lines = {
    # "1 week": 1,
    "1 month": 52.25 / 12,
    "1 year": 52.25,
    "5 years": 5 * 52.25,
    "30 years": 30 * 52.25,
    # "60 years": 60 * 52.25,
}

for ax, (key, simulation) in zip(axes.ravel()[:3], simulations.items()):
    for line_label, value in lines.items():
        va = "top"
        y = 0.95

        if key == "mu":
            if line_label == "1 month":
                continue
            else:
                va = "bottom"
                y = 0.05
        elif key == "rho":
            if line_label == "1 month":
                va = "bottom"
                y = 0.05
            elif line_label == "30 years":
                va = "center"
                y = 0.5
        elif key == "sigma":
            if line_label == "1 month":
                va = "center"
                y = 0.5
            else:
                va = "bottom"
                y = 0.05

        ax.axvline(value, ls=":", color="silver", zorder=0)
        ax.text(
            value,
            y,
            line_label,
            rotation=90,
            ha="center",
            va=va,
            color="silver",
            fontsize="small",
            transform=ax.get_xaxis_transform(),
            bbox={"fc": "w", "ec": "none"},
            zorder=0.5,
        )


for ax, (key, simulation) in zip(axes.ravel()[:3], simulations.items()):
    params = simulation["params"]
    x = params[key]
    x = (1 - x) / x
    y1 = simulation["summaries"]["frac_paired"][:, 0]
    ax.scatter(x, y1, marker=".", alpha=0.5, edgecolor="none", label="simulation")

    factor = 2
    lin = np.geomspace(x.min() / factor, x.max() * factor, 500)
    params_with_lin = params | {key: 1 / (1 + lin)}

    y2 = estimate_paired_fraction(params_with_lin["rho"], params_with_lin["mu"], params_with_lin["sigma"])
    # idx = np.argsort(x)
    ax.plot(lin, y2, color="k", ls="--", label="thermodynamic\nlimit")

    x = (1 - reference[key]) / reference[key]
    ax.axvline(x, color="C2", ls="-.")

    ax.set_xscale("log")
    ax.set_xlabel("$\\frac{1 - \\%s}{\\%s}$ (weeks)" % (key, key))


keys = ["all-fixed-500", "all-fixed-5000"]
for key in keys:
    summaries = simulations[key]["summaries"]
    frac_paired = np.asarray(summaries["_frac_paired"])[..., 1]
    n_lags = frac_paired.shape[1]
    t = np.arange(n_lags)
    loc = frac_paired.mean(axis=0)
    scale = 1.96 * frac_paired.std(axis=0) / np.sqrt(n_lags - 1)

    ax = axes[1, 1]
    line, = ax.plot(t, loc, ls="-")
    ax.fill_between(t, loc - scale, loc + scale, color=line.get_color(), alpha=0.2)
    # ax.axhline(estimate_paired_fraction(**all_fixed))
    ax.plot(t, estimate_paired_fraction(**all_fixed, lag=t), color="k", ls="--")
ax.set_xlabel("lag $\\tau$ (weeks)")

for i, ax in enumerate(axes.ravel()):
    if i < 3:
        ax.set_ylabel("fraction paired $f$")
    else:
        ax.set_ylabel("fraction paired $g_\\tau$")
    if i == 0:
        ha = "right"
        x = 0.95
    else:
        ha = "left"
        x = 0.05
    ax.text(x, 0.95, f"({'abcd'[i]})", transform=ax.transAxes, va="top", ha=ha)

ax.yaxis.set_ticks([0.5, 0.55, 0.6])

for year in [1, 2, 3, 4]:
    value = 52.25 * year
    ax.axvline(value, ls=":", color="silver", zorder=0)
    line_label = "1 year" if year == 1 else f"{year} years"
    ax.text(
        value,
        0.05,
        line_label,
        rotation=90,
        ha="center",
        va="bottom",
        color="silver",
        fontsize="small",
        transform=ax.get_xaxis_transform(),
        bbox={"fc": "w", "ec": "none"},
        zorder=0.5,
    )

fig.legend(
    handles=[
        mpl.lines.Line2D([], [], marker=".", ls="-"),
        mpl.lines.Line2D([], [], marker=".", ls="-", color="C1"),
        mpl.lines.Line2D([], [], color="k", ls="--"),
        mpl.lines.Line2D([], [], color="C2", ls="-."),
    ],
    labels=["simulation ($n=500$)", "simulation ($n=5{,}000$)", "thermodynamic limit", "Hannsson et al. (2019)"],
    fontsize="small",
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.075)
)
fig.tight_layout()
fig.savefig("thermodynamic-limit.pdf", bbox_inches="tight")
```

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

This notebook compares the RMSE for inference using different summary statistics:

- fractions only with longitudinal summaries
- summaries reported by Hansson et al. (2019)
- both of the above

It also plots priors for the different parameters and the corresponding posterior for inference using the summary statistics reported by Hansson et al. (2019). Further, posterior predictive replication of summary statistics is shown.

```{code-cell} ipython3
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
from hiv.util import transform_proba_continuous, transform_proba_discrete
from hiv.simulator import estimate_paired_fraction
from tqdm.notebook import tqdm
import collectiontools
```

```{code-cell} ipython3
def plot_rmse(filename, axes=None, **kwargs):
    filename = pathlib.Path(filename)
    result = pd.read_pickle(filename)

    if axes is None:
        fig, axes = plt.subplots(2, 3)
    else:
        fig = None

    for i, (ax, ys, name) in enumerate(zip(axes.ravel(), result["mses"].transpose(), result["param_names"])):

        # MSE ro RMSE.
        ys = ys ** 0.5
        # Average and standard error across simulations.
        # m = np.median(ys, axis=0)
        m = ys.mean(axis=0)
        s = ys.std(axis=0) / np.sqrt(ys.shape[0] - 1)
        x = np.arange(m.size)
        line, = ax.plot(x, m, **kwargs, alpha=0.7)
        ax.fill_between(x, m - s, m + s, color=line.get_color(), alpha=0.2)

        title = f"({'abcdef'[i]}) $\\{name}$".replace("0", "_0").replace("1", "_1")
        ax.set_title(title)

        formatter = ax.yaxis.major.formatter
        formatter.set_useMathText(True)
        formatter.set_powerlimits((0, 0))

    for ax in axes[1]:
        ax.set_xlabel("lag (weeks)")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE")

    if fig:
        fig.suptitle(filename)
        fig.tight_layout()
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3)
plot_rmse(
    "../workspace/default/result-adjusted-none-frac-only.pkl",
    axes,
    label="longitudinal",
)
plot_rmse(
    "../workspace/default/result-adjusted-none-hansson-only.pkl",
    axes,
    label="ideal diary",
    ls="--",
)
plot_rmse(
    "../workspace/default/result-adjusted-none-all.pkl",
    axes,
    label="both",
    ls=":",
)
axes[0, 0].legend(fontsize="small")
fig.tight_layout()
fig.savefig("longitudinal-rmse.pdf")
```

```{code-cell} ipython3
# Load a few batches of simulated data and stack them into "sims" for plotting.
simulations = [
    pd.read_pickle(f"../workspace/default/train/{i}.pkl")
    for i in tqdm(range(20))
]

sims = {}
for sim in simulations:
    collectiontools.append_values(sims.setdefault("params", {}), sim["params"])
    collectiontools.append_values(
        sims.setdefault("summaries", {}),
        {k: v for k, v in sim["summaries"].items() if not k.startswith("_")}
    )

sims["params"] = collectiontools.map_values(np.concatenate, sims["params"])
sims["summaries"] = collectiontools.map_values(np.concatenate, sims["summaries"])
sims["priors"] = simulations[0]["priors"]
```

```{code-cell} ipython3
inference = pd.read_pickle("../workspace/empirical/samples/default.pkl")
data = pd.read_pickle("../workspace/empirical/hansson2019data.pkl")
samples = dict(zip(inference["param_names"], inference["samples"].squeeze().T))
samples.keys()
```

```{code-cell} ipython3
# Run the posterior predictive checks first.
# "features" from the data.
summaries = {
    key: value.squeeze()
    for key, value in data["summaries"].items()
    if np.isfinite(value).all()
}
ppd_summaries = dict(zip(inference["feature_names"], inference["features"].squeeze().T))
prior_features = {
    key: value[..., 0] for key, value
    in sims["summaries"].items()
    if not key.startswith("_")
}

labels = {
    "frac_paired": "fraction paired",
    "frac_concurrent": "fraction with\nconcurrent partners",
    "steady_length": "length of steady\nrelationships (days)",
    "casual_gap_single": "time to new casual\npartner, single (days)",
    "casual_gap_paired": "time to new casual\npartner, paired (days)",
}

fig, axes = plt.subplots(2, 3)
axes = axes.ravel()

for i, (key, value) in enumerate(summaries.items()):
    ax = axes[i]
    # Plot histograms.
    x = ppd_summaries[key]
    y = prior_features[key]

    if key in {"casual_gap_single", "casual_gap_paired", "steady_length"}:
        x = x * 365
        y = y * 365
        value = value * 365

    xy = np.concatenate([x, y])
    xmin, xmax = np.quantile(xy, [0.005, 0.995])
    kwargs = {"density": True, "range": (xmin, xmax), "bins": 20}
    ax.hist(x, **kwargs, label="posterior predictive")
    ax.hist(y, alpha=0.5, **kwargs, label="prior predictive")

    # Show observations.
    ax.axvline(value, color="k", ls="--", label="observed")
    ax.set_xlabel(labels[key])
    ax.set_yticks([])

lax = axes[-1]
lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc="center")

for ax in axes[::3]:
    ax.set_ylabel("density")

for i, ax in enumerate(axes[:5]):
    label = "abcdef"[i]
    ax.text(0.95, 0.95, f"({label})", transform=ax.transAxes, va="top", ha="right")

fig.tight_layout()
fig.savefig("posterior-predictive-replication.pdf")
```

```{code-cell} ipython3
fig = plt.figure()
gs = fig.add_gridspec(2, 3)

priors = {
    key: value for key, value in sims["priors"].items()
    if not isinstance(value, (int, float))
}

lines = {
    "1 week": 1,
    "1 month": 52.25 / 12,
    "1 year": 52.25,
    "5 years": 5 * 52.25,
    "30 years": 30 * 52.25,
}

estimates = {
    # They work on {{{HIV in MSM population}}}.
    "Hansson et al. (2019)": {
        # By assumption.
        "mu": transform_proba_continuous(1 / 60, 7 / 365.25),
        # From first paragraph, second column, page 70.
        "rho": transform_proba_continuous(1 / 163, 7 / 1),
        # Based on first line of table 2 and assumption of mu.
        "sigma": transform_proba_continuous(0.003333395841, 7 / 1),
        # Based on second column on page 70.
        "omega_0": transform_proba_continuous(1 / 23.1, 7),
        "omega_1": transform_proba_continuous(1 / 36.3, 7),
    },
    # An early stochastic pair formation model. Parameters are "biologically plausible"
    # but not data-informed. They were extracted from Table 1 on page 182. All
    # simulations were run at a resolution of one day in discrete time. We don't use it
    # here because it's the same parameters as in the 1998 paper but without `mu`.
    # "Kretzschmar et al. (1996)": {
    #    "rho": transform_proba_discrete(0.01, 7),
    #    "sigma": transform_proba_discrete(0.005, 7),
    # },

    # A continuous-time ODE model for pair formation. They use a recruitment rate
    # $\nu$ instead of the expected population size $n$, but they are equivalent
    # because $n = \nu / \mu$ for emmigration rate $\mu$. Parameters were extracted
    # from the caption of Fig. 1, 2, 3, 4, 5. There is sensitivity analysis to
    # rho = [0.365, 1.095, 1.825, 3.65, 10.95, 18.25] / year, but rho = 3.65 / year
    # is most commonly used. Fig. {6, 7} note rho = {0.05, 0.01} / day, but those
    # values are equivalent to the yearly values reported in earlier figures. They focus
    # on {{{HIV in MSM population}}}.
    "Kretzschmar et al. (1998)": {
        "mu": transform_proba_continuous(0.1095, 7 / 365.25),
        "rho": transform_proba_continuous(3.65, 7 / 365.25),
        "sigma": transform_proba_continuous(1.825, 7 / 365.25),
    },
    # Study in Amsterdam with interviews of young homosexual men (<= 30 years) in
    # Amsterdam. They use a continuous-time model as discussed in the appendix, and
    # parameter values are shown in table 1. There is also a set of priors for
    # sensitivity analysis shown in table 2. They focus on {{{HIV in MSM population}}}.
    "Xiridou et al. (2003)": {
        # They call this rho_s for singles acquiring casual partners.
        "omega_0": transform_proba_continuous(22, 7 / 365.25),
        # They call this rho_m for paired nodes acquiring casual partners.
        "omega_1": transform_proba_continuous(8, 7 / 365.25),
        "rho": transform_proba_continuous(0.73, 7 / 365.25),
        "sigma": transform_proba_continuous(1 / 1.5, 7 / 365.25),
        "mu": transform_proba_continuous(1 / 30, 7 / 365.25),
    },
    # Study on HPV with parameters from table 2. They consider {{{HPV in a heterosexual
    # population}}}.
    "Saldana et al. (2025)": {
        "mu": transform_proba_continuous(1 / 9, 7 / 365.25),
        "rho": transform_proba_continuous(5, 7 / 365.25),
        "sigma": transform_proba_continuous(2, 7 / 365.25),
    },
    # They consider a continuous-time pair-formation model. Parameter values are
    # largely from figure 1 and section 3.1, and rates are on a yearly scale. They
    # mention parameter inference in section 3.1. Parameters are estimated based on
    # matching to summary statistics from a survey. It probably is worth verifying
    # that these values match. Unlike the other models, the parameter estimation is
    # for heterosexual partnerships while the model itself does not have any
    # male/female structure. The data are from AddHealth. There is some variation in
    # kappa and K further down in the paper (around Fig. 3) as a sensitivity analysis.
    # They consider {{{chlamydia, gonorrhea, hpv in heterosexual population}}}.
    "Leng et al. (2018)": {
        # They assume a closed population.
        "mu": 0,
        # They call this $f$.
        "rho": transform_proba_continuous(3, 7 / 365.25),
        # They use parameter $b$ to denote the rate at which *one* partner breaks up
        # the relationship, so the rate of dissolution is $2b$. This setup matches their
        # summary statistics of 2 / 3 of individuals having an exclusive relationship.
        "sigma": transform_proba_continuous(1.5, 7 / 365.25),
        # They call this $\kappa$.
        "omega_0": transform_proba_continuous(2 * 0.335, 7 / 365.25),
        # They call this $K$. In their simulations, they set $\kappa = 2 K$ to reflect
        # that singles have a higher rate of casual contacts than non-singles.
        "omega_1": transform_proba_continuous(0.335, 7 / 365.25),
    },
    # Recent book chapter. They consider {{{HIV and HSV-2 in a unipartite population}}}.
    "Gurski et al. (2025)": {
        "mu": transform_proba_continuous(1 / 61, 7 / 365.25),
        "rho": transform_proba_continuous(5, 7 / 365.25),
        "sigma": transform_proba_continuous(2.11, 7 / 365.25),
    },
    # Betti would be interesting to include here, but they consider general
    # transmission, not within the MSM community and thus have very different parameter
    # estimates.
}

use_log = True

ax = None
axes = []
for i, (name, prior) in enumerate(priors.items()):
    xs = samples[name].clip(1e-6, 1)

    # Report the raw probabilities.
    print(f"### {name} ###")
    print(f"proba: {name}", np.quantile(xs, [0.5, 0.025, 0.975]))

    # We want to plot the probabilities on the beta-prime scale because that's much more
    # interpretable. `xi` is an exception because it doesn't represent a notion of time.
    if name != "xi":
        ax = fig.add_subplot(gs[i // 3, i % 3], sharex=ax)
        prime = stats.betaprime(*prior.args[::-1])
        f = 0.001
        lower, upper = prime.ppf([f, 1 - f])
        x = np.geomspace(lower, upper)
        pdf = prime.pdf(x)
        if use_log:
            pdf = pdf * x
            ax.set_xscale("log")
        for line_label, value in lines.items():
            ax.axvline(value, ls=":", color="silver")
            ax.text(
                value,
                .95,
                line_label,
                rotation=90,
                ha="center",
                va="top",
                color="silver",
                fontsize="small",
                transform=ax.get_xaxis_transform(),
                bbox={"fc": "w", "ec": "none"},
                zorder=2,
            )

        name = name.replace("0", "_0").replace("1", "_1")
        label = fr"$\frac{{1-\{name}}}{{\{name}}}$ (weeks)"

        xs = (1 - xs) / xs
        print(f"time: {name}", np.quantile(xs, [0.5, 0.025, 0.975]))
    else:
        ax = fig.add_subplot(gs[-1, -1])
        label = r"$\xi$"
        x = np.linspace(0, 1)
        pdf = prior.pdf(x)

    # Plot the prior.
    ax.plot(x, pdf, color="w", lw=5, zorder=2.5)
    ax.plot(x, pdf, color="k", zorder=2.5, label="prior", ls="--")

    tax = ax.twinx()
    axes.append(ax)
    axes.append(tax)

    # Plot the posterior.
    hist_kwargs = {
        "color": "C0",
        "alpha": 0.5,
    }

    f = 0.1
    if name == "xi":
        xmin, xmax = xs.min(), xs.max()
        span = xmax - xmin
        x = np.linspace(xmin - f * span, xmax + f * span, 200)

        pdf = stats.gaussian_kde(xs)(x)
    else:
        log_xs = np.log10(xs)
        xmin, xmax = log_xs.min(), log_xs.max()
        span = xmax - xmin
        x = np.logspace(xmin - f * span, xmax + f * span, 200)
        pdf = stats.gaussian_kde(xs)(x) * x
    # tax.plot(x, pdf, color="w", lw=5, zorder=2.5)
    tax.plot(x, pdf, label="posterior")

    ax.set_xlabel(label)

    # Align the baseline.
    ax.set_ylim(0)
    tax.set_ylim(0)
    ax.set_yticks([])
    tax.set_yticks([])

    if i % 3 == 0:
         ax.set_ylabel("density")
    # if i % 3 == 2:
    #     tax.set_ylabel("posterior density", color="C0")

    # Add the estimates.
    for i, (label, params) in enumerate(estimates.items()):
        param = params.get(name)
        if param is None or param == 0:
            continue
        param = (1 - param) / param
        tax.scatter(param, 0, facecolor=f"C{i + 1}", zorder=999, marker=10, clip_on=False, alpha=0.5, label=label)


for i, ax in enumerate(axes[::2]):
    label = "abcdef"[i]
    ax.text(0.95, 0.95, f"({label})", transform=ax.transAxes, va="top", ha="right")

# Add a legend.
labels_and_handles = {}
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        labels_and_handles[label] = handle
handles_and_labels = tuple(zip(*(item[::-1] for item in labels_and_handles.items())))
fig.legend(*handles_and_labels, fontsize="small", ncol=3, loc="center", bbox_to_anchor=(.5, 1.05))

fig.tight_layout()
fig.savefig("parameters.pdf", bbox_inches="tight")
```

```{code-cell} ipython3
pd.DataFrame(estimates)
```

```{code-cell} ipython3
for key, estimate in estimates.items():
    frac = estimate_paired_fraction(estimate["rho"], estimate["mu"], estimate["sigma"])
    print(f"{key}: {frac:.2f}")
```

---
jupytext:
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
import enum
from matplotlib import pyplot as plt
import numbers
import numpy as np
from scipy import optimize, stats
```

```{code-cell} ipython3
class Unit(enum.Enum):
    # Days etc have units of time. Daily etc. have units of inverse time.
    UNITLESS = 0
    DAYS = 1
    DAILY = 2
    WEEKS = 3
    WEEKLY = 4
    YEARS = 5
    YEARLY = 6
```

```{code-cell} ipython3
raw = {
    # Extracted from table 1.
    "kretzschmar1996": {
        "n": (2000, Unit.UNITLESS),
        "rho": (0.01, Unit.DAILY),
        "sigma": (0.005, Unit.DAILY),
        "xi": (np.linspace(0, 1, 5), Unit.UNITLESS),
        "omega0": None,
        "omega1": None,
        "mu": None,
    },
    #
    "kretzschmar1998": {
        # From figure 2.
        "n": (1000, Unit.UNITLESS),
        # From figure 1 and 2.
        "mu": (0.1095, Unit.YEARLY),
        # From figure 2. Figure 1 considers a range from 0.365 to 18.25.
        "rho": (3.65, Unit.YEARLY),
        # From figure 2. In figure 1, sigma is implicitly determined through
        # `duration = 1 / (sigma + 2 mu)` such that `sigma = 1 / duration - 2 mu`
        # with `duration` between 0 and 2. For consistency with the other parameters.
        "sigma": (0.1825, Unit.YEARLY),
        "xi": None,
        "omega0": None,
        "omega1": None,
    },
    # Extracted from table 1.
    "xiridou2003": {
        "n": (20000, Unit.UNITLESS),
        "omega0": ([16, 22, 28], Unit.YEARLY),
        "omega1": ([6, 8, 10], Unit.YEARLY),
        "rho": (0.73, Unit.YEARLY),
        "sigma": ([0.75, 1.5, 2.5], Unit.YEARS),
        "mu": (30, Unit.YEARS),
        "xi": None,
    },
    #
    "leng2018": {
        "n": None,
        "mu": None,
        "rho": (3, Unit.YEARLY),
        "sigma": (1.5, Unit.YEARLY),
        "omega1": (0.335, Unit.YEARLY),
        "omega0": (2 * 0.335, Unit.YEARLY),
        "xi": None,
    },

    "hansson2019": {
        "n": (5000, Unit.UNITLESS),
        "mu": (60, Unit.YEARS),
        "rho": (58.7, Unit.DAYS),
        "sigma": (296, Unit.DAYS),
        "omega1": (36, Unit.DAYS),
        "omega0": (23, Unit.DAYS),
        "xi": None,
    },
}
raw
```

```{code-cell} ipython3
discrete_models = {"kretzschmar1996"}
weeks_per_year = 52.1775
weeks_per_day = 1 / 7

# Transform all parameters to weekly probabilities.
transformed = {}
for model_key, raw_params in raw.items():
    transformed_params = {}
    for key, value in raw_params.items():
        # No parameters.
        if value is None:
            transformed_params[key] = None
            continue

        # Unitless parameters.
        value, unit = value
        if not isinstance(value, numbers.Number):
            value = np.asarray(value)
        if unit == Unit.UNITLESS:
            transformed_params[key] = value
            continue

        if model_key in discrete_models:
            match unit:
                case Unit.DAILY:
                    value = 1 - (1 - value) ** 7
                case _:
                    raise NotImplementedError(unit)
        else:
            match unit:
                case Unit.YEARLY:
                    # The expected number of weekly occurrences is equal to the
                    # expected number of yearly occurrences divided by the number
                    # of weeks per year.
                    value = value / weeks_per_year
                case Unit.YEARS:
                    # In discrete-time models, durations (such as the length of
                    # relationships) are modeled using geometric distributions.
                    # We are interested in the number of trials until the first
                    # "success" (such as a relationship breaking up). Following
                    # https://en.wikipedia.org/wiki/Geometric_distribution,
                    # the expected duration until the first success is 1 / p for
                    # success probability p. Thus, the probability is one over
                    # the expected value converted to weeks.
                    value = 1 / (value * weeks_per_year)
                case Unit.DAYS:
                    value = 1 / (value * weeks_per_day)
                case _:
                    raise NotImplementedError(unit)

        if np.max(value) > 0.1:
            print(model_key, key, value)
        transformed_params[key] = value

    assert len(transformed_params) == 7, model_key
    transformed[model_key] = transformed_params

transformed
```

```{code-cell} ipython3
def ms2hyperparams(mu, sigma):
    # From https://en.wikipedia.org/wiki/Beta_prime_distribution.
    sigma2 = sigma * sigma
    a = (mu * (mu + mu**2 + sigma2))/sigma2
    b = 2 + (mu * (1 + mu))/sigma2
    return a, b


def qq2hyperparams(x1, q1, x2, q2):
    result = optimize.minimize(
        lambda x: np.linalg.norm([q1, q2] - stats.betaprime(*np.exp(x)).cdf([x1, x2])),
        np.zeros(2),
        method="powell",
    )
    params = np.exp(result.x)
    np.testing.assert_allclose(stats.betaprime(*params).cdf([x1, x2]), [q1, q2], rtol=1e-2)
    return params


hyperparams = {
    # "mu": ms2hyperparams(25 * weeks_per_year, 15 * weeks_per_year),
    "mu": qq2hyperparams(5 * weeks_per_year, 0.05, 60 * weeks_per_year, 0.95),
    # "rho": ms2hyperparams(15, 20),
    "rho": qq2hyperparams(2, 0.05, 2 * weeks_per_year, 0.95),
    "sigma": qq2hyperparams(10, 0.05, 10 * weeks_per_year, 0.95),
    "omega0": qq2hyperparams(.5, 0.05, 8, 0.95),
    "omega1": qq2hyperparams(2, 0.05, 20, 0.95),
    "xi": (2, 2),
}
colors = {
    "kretzschmar1996": "C0",
    "kretzschmar1998": "C1",
    "xiridou2003": "C2",
    "leng2018": "C3",
    "hansson2019": "C4",
}
use_log_scale = True

fig = plt.figure()
gs = fig.add_gridspec(3, 3, height_ratios=[.01, 1, 1])
lax = fig.add_subplot(gs[0, :])

axes = [fig.add_subplot(gs[i, j]) for i in range(1, 3) for j in range(3)]
for ax in axes[1:-1]:
    ax.sharex(axes[0])

# Transpose and plot the parameters.
for model_key, transformed_params in transformed.items():
    sorted_filtered = sorted(x for x in transformed_params.items() if x[0] != "n")
    for ax, (key, value) in zip(axes, sorted_filtered):
        if value is None:
            continue

        _, labels = ax.get_legend_handles_labels()
        plotted = "prior" in labels

        if key == "xi":
            param_label = f"$\\{key}$"
            dist = stats.beta(*hyperparams[key])
            x = np.linspace(0, 1, 1000)
            pdf = dist.pdf(x)
        else:
            # This is the expected number of weeks with "failures" (i.e. no event happening)
            # before the first success (i.e. an event happening, such as breaking a relationship).
            value = (1 - value) / value
            # TODO: fix labels here.
            param_label = f"$\\frac{{1 - \\{key}}}{{\\{key}}}$ (weeks)"
            dist = stats.betaprime(*hyperparams[key])
            x = np.linspace(*dist.ppf([0.01, 0.99]), 1000)
            pdf = dist.pdf(x)

            if use_log_scale:
                ax.set_xscale("log")
                pdf *= x

            lines = {
                "1 week": 1,
                "1 month": weeks_per_year / 12,
                "1 year": weeks_per_year,
                "5 years": 5 * weeks_per_year,
                "30 years": 30 * weeks_per_year,
            }
            for line_label, line_value in lines.items():
                ax.axvline(line_value, ls=":", color="silver", zorder=0)
                if not plotted:
                    ax.text(
                        line_value, pdf.max(), line_label,
                        rotation=90, color="silver", fontsize="small", ha="center", va="top",
                        backgroundcolor="w", zorder=1
                    )

        if not plotted:
            # White line to separate the year, month labels from the prior line.
            ax.plot(x, pdf, color="w", lw=5)
            ax.plot(x, pdf, color="k", label="prior")
        ax.scatter(
            value,
            pdf.max() * 0.02 * np.random.uniform(-1, 1, np.shape(value)),
            label=model_key,
            marker="|",
            color=colors[model_key],
        )
        ax.set_xlabel(param_label)
        ax.yaxis.major.formatter.set_useMathText(True)
        ax.yaxis.major.formatter.set_powerlimits((0, 0))

# Gather all the artists and deduplicate by model key.
labels_handles = {}
for ax in axes:
    labels_handles.update({label: handle for handle, label in zip(*ax.get_legend_handles_labels())})
handles, labels = zip(*((handle, label) for label, handle in labels_handles.items()))
lax.legend(handles, labels, fontsize="small", ncol=4)
lax.set_axis_off()


# Note that Leng consider heterosexual relationships in schools, i.e. the add health data to
# inform their parameters. These can be quite different from what we're interested in. Specifically,
# they use the fraction of reported concurrency to inform the omega's, but that statistic is likely
# better suited to inform xi because the reported concurrency isn't casual interactions.

fig.tight_layout()
```

```{code-cell} ipython3
{key: tuple(map(lambda x: round(x, 4), value[::-1])) for key, value in hyperparams.items()}
```

```{code-cell} ipython3
{key: stats.beta(*value[::-1]).mean() for key, value in hyperparams.items()}
```

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

Various publications use models that are special cases of the universal simulator we are considering here. To validate the simulator, we can simulate for those specific cases and then compare summary statistics with the theoretical results published in the publications.

```{code-cell} ipython3
from hiv.simulator import UniversalSimulator
from hiv.scripts.generate_data import __main__ as _generate_data
from matplotlib import pyplot as plt
import numpy as np
import pickle
from unittest.mock import patch
```

```{code-cell} ipython3
def generate_data(args, **params):
    """
    Generate data using the generator script but return the results
    instead of writing to disk.
    """
    result = []
    args = [
        f"--param={key}={value}" for key, value in params.items()
    ] + list(map(str, args)) + ["--save-graphs", "dummy.pkl"]
    with patch("pickle.dump", lambda x, _: result.append(x)):
        _generate_data(args)
    result, = result
    return result


result = generate_data([10, 10], n=100)
```

# Kretzschmar and Heijne (2017)

The review article of Kretzschmar and Heijne (2017) considers a simple pair formation model with formation rate $\rho$ and dissolution rate $\sigma$, immigration rate $B=n/\mu$, and emigration rate $\mu$. We have $\omega_0=\omega_1=\xi=0$ and use small parameters for all probabilities to approximate the continuous dynamics well. We do set values for the $\omega$ because they don't affect the dynamics of steady relationships.

```{code-cell} ipython3
rho = 0.06
sigma = 0.03
mu = 0.01
omega0 = 0.2
omega1 = 0.1

result = generate_data(
    [200, 1, "--burnin=100"],
    n=1000,
    xi=0,
    rho=rho,
    sigma=sigma,
    mu=mu,
    omega0=omega0,
    omega1=omega1,
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2)

# Fraction of individuals in relationship from inline text about 3/4 
# of the way down page 369.
ax = axes[0, 0]
frac, = result["summaries"]["frac_paired"].T
ax.hist(frac)
ax.axvline(rho / (rho + sigma + 2 * mu), color="k")

# Fraction of singles with steady relationship.
ax = axes[0, 1]
frac, = result["summaries"]["frac_single_with_casual"].T
ax.hist(frac)
ax.axvline(omega0, color="k")

# Fraction of non-singles with steady relationship.
ax = axes[1, 0]
frac, = result["summaries"]["frac_paired_with_casual"].T
ax.hist(frac)
ax.axvline(omega1, color="k")
```

## Free Parameters, keeping $\xi=0$

We also let $\mu$, $\sigma$, and $\rho$ vary to check if we can recover the properties over a wide range of parameters. We keep $\xi=0$ to enforce serial monogamy.

```{code-cell} ipython3
result = generate_data(
    [200, 1, "--burnin=100"],
    n=1000,
    xi=0,
)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
rho = result["params"]["rho"]
mu = result["params"]["mu"]
sigma = result["params"]["sigma"]
ax.scatter(
    rho / (rho + sigma + 2 * mu),
    result["summaries"]["frac_paired"][:, 0],
)
ax.set_xlabel("theory")
ax.set_ylabel("empirical")
ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), color="k")
fig.tight_layout()
```

## Free parameters

Finally, we free up all parameters and check to what extent the above relation still holds. The number of paired nodes is slightly biased upwards as we'd expect given that more parameters can participate in pair formation.

```{code-cell} ipython3
result = generate_data(
    [200, 1, "--burnin=100"],
    n=1000,
)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
rho = result["params"]["rho"]
mu = result["params"]["mu"]
sigma = result["params"]["sigma"]
ax.scatter(
    rho / (rho + sigma + 2 * mu),
    result["summaries"]["frac_paired"][:, 0],
    c=result["params"]["xi"],
)
ax.set_xlabel("theory")
ax.set_ylabel("empirical")
ax.set_aspect("equal")
ax.plot((0, 1), (0, 1), color="k")
fig.tight_layout()
```

## Diagnosing $\xi$ Sensitivity

Intuitively, the fraction of nodes with concurrent relationships should correlate with $\xi$, but there is no obvious correlation when scattering the fraction of nodes with concurrent connections and $\xi$. That is, at least in part, due to concurrent connections being very sensitive to $\sigma$ because connections dissolve independently. Here, we eliminate variability due to different parameters one-by-one and determine when $\xi$ becomes informative of the fraction of nodes with concurrent relationships. We find that concurrency is *very* sensitive to both $\rho$ and $\sigma$.

```{code-cell} ipython3
result = generate_data(
    [100, 1, "--burnin=200"],
    n=2000,
    rho=0.01,
    sigma=0.005,
    mu=0.001,
    omega0=0,
    omega1=0,
)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(
    result["params"]["xi"], 
    result["summaries"]["frac_concurrent"],
)
```

# Kretzschmar and Morris (1996)

They expand the above model with an additional mixing function which allows for individuals to form more than one steady relationship at a time, but have a closed population ($\mu=0$) and no casual relationships ($\omega_0=\omega_1=0$). In particular, the probability to connect is $\rho$ for singles and $\rho\xi$ for non-singles. They use $1-\xi$ instead of $\xi$ but the parameterization in terms of $\xi$ is nicer. See page 180 for details. On page 184, they report the degree distribution for different values of $\xi$ which we can try to reproduce here for just a few values. For all other parameters, we use the values reported in table 1 on page 182.

```{code-cell} ipython3
rho = 0.01
sigma = 0.005
xis = np.linspace(0, 1, 10)
mu = 0

degree_dists = []
mean_degrees = []
frac_concurrent = []

for xi in xis:
    result = generate_data(
        [100, 1, "--burnin=200"],
        n=2000,
        xi=xi,
        rho=rho,
        sigma=sigma,
        mu=mu,
        omega0=omega0,
        omega1=omega1,
    )
    
    degrees = []
    for graph, in result["graph_sequences"]:
        degrees.append(graph.degrees("steady"))
    degrees = np.ravel(degrees)
    degree_dist = np.bincount(degrees) / degrees.size
    degree_dists.append(degree_dist)
    mean_degrees.append(degrees.mean())
    frac_concurrent.append(result["summaries"]["frac_concurrent"].mean())
    
print("mean degrees", np.asarray(mean_degrees))
```

We cannot quite reproduce the statistics reported in the table because the authors limit the number of connections to explicitly consider the effect of concurrency without confounding by the total number of connections. However, we do recover the correct values for the unambiguous case of serial monogamy, i.e., $\xi=0$ in our parameterization and $\xi=1$ in theirs.

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2)
for degree_dist in degree_dists:
    ax1.plot(degree_dist, marker="o")
ax1.set_xlabel("degree")
ax1.set_ylabel("degree density")
    
ax2.plot(xis, frac_concurrent)
ax2.set_xlabel("xi")
ax2.set_ylabel("fraction of nodes with concurrency")
fig.tight_layout()
```

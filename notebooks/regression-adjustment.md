---
jupytext:
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
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
```

```{code-cell} ipython3
n_samples = 100_000
nu = 1
kappa = 2
sigma = 0.3
x_obs = np.asarray([-2])
frac = 0.05

np.random.seed(0)
mus = np.random.normal(nu, kappa, (n_samples, 1))
xs = np.random.normal(mus, sigma)

dists = np.abs(xs - x_obs).squeeze()
idx = np.argsort(dists)[:int(frac * n_samples)]
mu_samples = mus[idx]
x_samples = xs[idx]
```

```{code-cell} ipython3
model = LinearRegression().fit(x_samples, mu_samples)
correction = model.predict(x_obs[None, :])[0] - model.predict(x_samples)
mu_samples_corrected = mu_samples + correction
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(xs[::10], mus[::10])
ax.axvline(x_obs, color="k", ls="--")
ax.scatter(x_samples, mu_samples)

fig.tight_layout()
```

```{code-cell} ipython3
fig, ax = plt.subplots()

posterior_precision = 1 / kappa ** 2 + 1 / sigma ** 2
posterior_mean = (nu / kappa ** 2 + x_obs / sigma ** 2) / posterior_precision

posterior = stats.norm(posterior_mean, 1 / np.sqrt(posterior_precision))
lin = np.linspace(*posterior.ppf([0.01, 0.99]))
ax.plot(lin, posterior.pdf(lin))
ax.plot(lin, stats.gaussian_kde(mu_samples.T)(lin))
ax.plot(lin, stats.gaussian_kde(mu_samples_corrected.T)(lin))
# ax.hist(mu_samples, density=True, bins=21)
# ax.hist(mu_samples_corrected, density=True, bins=21)
```

```{code-cell} ipython3
plt.scatter(x_samples, correction)
```

# HIV Sexual Contact Network Simulation [![CI](https://github.com/onnela-lab/hiv/actions/workflows/main.yml/badge.svg)](https://github.com/onnela-lab/hiv/actions/workflows/main.yml)

Longitudinal inference for sexual contact networks using simulation-based methods. This package implements a universal discrete-time simulator for sexual contact networks and performs approximate Bayesian computation (ABC) inference.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. Then clone the repository and install dependencies by running:

```bash
uv sync --all-groups
```

## Reproducing Results

To reproduce the results in accompanying publication, run the following commands. All simulations and ABC are orchestrated using the `cook` task runner. The following command will generate all results required to create the figures.

```bash
uv run cook exec -j 4 "default/inference/adjusted/none/*" empirical/inference/default
```

It will take a while to run: 96 hours CPU clock time on a M4 MacBook Air or 24 hours wall clock time with four processes. You can modify the number of concurrent processes by changing the `-j [number of processes]` flag.

Launch a Jupyter notebook and navigate to the [`notebooks`](./notebooks/) directory which contains notebooks in [Jupytext](https://jupytext.readthedocs.io/en/latest/) markdown format. Right-click any notebook in the Jupyter explorer and select "Open With > Notebook". Executing the notebooks will generate the figures in the manuscript.

- [notebooks/illustration.md](notebooks/illustration.md): Illustration of the network evolution mechanisms (figure 1).
- [notebooks/thermodynamic-limit.md](notebooks/thermodynamic-limit.md): Validation of analytic expressions in the thermodynamic limit against finite-size simulations (figure 2). This notebook might take another hour or so to run.
- [notebooks/inference.md](notebooks/inference.md): Prior, posterior, and parameter values reported in the literature (figure 3), samples of summary statistics from the prior-predictive and posterior-predictive distributions (figure 4), and evolution of parameter RMSE as a function of between-wave follow-up period (figure 5).

## Background

The [`src/hiv`](src/hiv) directory contains the Python package responsible for simulation and inference. The main files are:

- [`algorithm.py`](src/hiv/algorithm.py): Implementation of nearest-neighbor ABC as a scikit-learn estimator and function to apply regression adjustment to candidate posterior samples.
- [`simulator.py`](src/hiv/simulator.py): Efficient sexual contact network simulator for unipartite graphs.
- [`util.py`](src/hiv/util.py): Utility function and implementation of `NumpyGraph`, a NumPy-based graph representation. In short, nodes and edges are stored as NumPy arrays which allows for performant batch-wise updates, e.g., deleting a set of nodes, adding a set of edges, etc. It is *very* slow for iterative updates.
- [`scripts`](src/hiv/scripts): Collection of command-line interfaces.j
   - [`generate_data.py`](src/hiv/scripts/generate_data.py): Generate samples from the prior predictive distribution of the sexual contact network model. This is primarily used for constructing the ABC reference table.
   - [`run_abc.py`](src/hiv/scripts/run_abc.py): Run approximate Bayesian computation for observed data given a reference table.

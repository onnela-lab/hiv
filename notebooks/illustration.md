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
import matplotlib as mpl
from matplotlib import pyplot as plt
from shapely import geometry
from scipy.spatial import ConvexHull
import numpy as np
import itertools as it
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
axes = axes.ravel()

titles = [
    "state at $t$",
    "node and edge deletion",
    "node addition",
    "steady edge addition",
    "casual contact addition",
    "new state at $t+1$",
]
assert len(titles) == 6

pos = {
    0: (0, 2),
    1: (0, 1),
    2: (1.5, 1),
    3: (3, 2),
    4: (4, 3),
    5: (2, 3),
    6: (2, 5),
    7: (1, 4),
    8: (0, 3),
}

original_nodes = set(range(7))
deleted_nodes = {1, 4}
new_nodes = {7, 8}

original_steady = {
    (2, 3),
    (3, 4),
    (5, 6),
}
deleted_steady = {
    (3, 4),
    (5, 6),
}
new_steady = {
    (0, 8),
    (3, 5),
}

original_casual = {
    (0, 1),
    (3, 5),
}
new_casual = {
    (0, 2),
    (7, 8),
}


deleted_kwargs = {
    "color": "#ddd",
    "zorder": 0,
}


def plot_nodes(pos, exclude=None, include=None, labels=False, ax=None, **kwargs):
    kwargs = {"zorder": 3, "edgecolor": "w", "s": 250, "clip_on": False} | kwargs
    ax = ax or plt.gca()

    assert not (include and exclude)

    if exclude:
        pos = {key: value for key, value in pos.items() if key not in exclude}
    if include:
        pos = {key: value for key, value in pos.items() if key in include}

    ax.scatter(*zip(*pos.values()), **kwargs)
    if labels:
        for node, xy in pos.items():
            ax.text(*xy, str(node))


def plot_edges(edges, pos, ax=None, **kwargs):
    kwargs = {
        "color": "k",
    } | kwargs
    ax = ax or plt.gca()
    for u, v in edges:
        nodes = (pos[u], pos[v])
        ax.plot(*zip(*nodes), **kwargs)


def label_node(xy, text, ax=None, **kwargs):
    ax = ax or plt.gca()
    kwargs = {"color": "w", "ha": "center", "va": "center"} | kwargs
    ax.text(*xy, text, **kwargs)


# The starting graph.
ax = axes[0]
nodes = original_nodes
plot_nodes(pos, include=nodes, ax=ax)
plot_edges(original_steady, pos, ax=ax)
plot_edges(original_casual, pos, ax=ax, ls="--")

# Deletion.
ax = axes[1]
nodes = original_nodes - deleted_nodes
steady = original_steady - deleted_steady

plot_nodes(pos, include=nodes, ax=ax)
plot_nodes(pos, include=deleted_nodes, ax=ax, color="C3")
for node in deleted_nodes:
    label_node(pos[node], "$\\mu$", ax=ax)

plot_edges(steady, pos, ax=ax)
plot_edges(deleted_steady, pos, ax=ax, color="C3")
plot_edges(original_casual, pos, ax=ax, color="C3", ls="--")

center = np.mean([pos[5], pos[6]], axis=0)
ax.text(
    *center, "$\\sigma$", color="C3", ha="center", va="center",
    bbox=dict(facecolor="w", edgecolor="none", alpha=0.9)
)

# Add nodes.
ax = axes[2]
plot_nodes(pos, include=nodes, ax=ax)
plot_nodes(pos, include=new_nodes, ax=ax, color="C2")
plot_edges(steady, pos, ax=ax)

for node in new_nodes:
    label_node(pos[node], "$n\\mu$", ax=ax, fontsize="small")

plot_nodes(pos, include=deleted_nodes, ax=ax, **deleted_kwargs)
plot_edges(deleted_steady, pos, ax=ax, **deleted_kwargs)
plot_edges(original_casual, pos, ax=ax, ls="--", **deleted_kwargs)

nodes = nodes | new_nodes

# Add steady edges.
ax = axes[3]
plot_nodes(pos, include=nodes, ax=ax)
plot_edges(steady, pos, ax=ax)
plot_edges(new_steady, pos, ax=ax, color="C2")
steady = steady | new_steady

plot_nodes(pos, include=deleted_nodes, ax=ax, **deleted_kwargs)
plot_edges(deleted_steady, pos, ax=ax, **deleted_kwargs)
plot_edges(original_casual, pos, ax=ax, ls="--", **deleted_kwargs)

hull = ConvexHull([pos[node] for edge in new_steady for node in edge])
poly = geometry.Polygon(hull.points[hull.vertices]).buffer(0.5)
poly = mpl.patches.Polygon(
    np.transpose(poly.boundary.xy),
    edgecolor="C2",
    facecolor=mpl.colors.to_rgba("C2", alpha=0.1),
)
ax.add_patch(poly)

for node in it.chain.from_iterable(new_steady):
    if node == 3:
        label = "$\\rho\\xi$"
        fontsize = "small"
    else:
        label = "$\\rho$"
        fontsize = None

    label_node(pos[node], label, ax=ax, fontsize=fontsize)


# Add casual edges.
ax = axes[4]
plot_nodes(pos, include=nodes, ax=ax)
plot_edges(steady, pos, ax=ax)
plot_edges(new_casual, pos, ax=ax, color="C2", ls="--")

plot_nodes(pos, include=deleted_nodes, ax=ax, **deleted_kwargs)
plot_edges(deleted_steady, pos, ax=ax, **deleted_kwargs)
plot_edges(original_casual, pos, ax=ax, ls="--", **deleted_kwargs)

hull = ConvexHull([pos[node] for edge in new_casual for node in edge])
poly = geometry.Polygon(hull.points[hull.vertices]).buffer(0.5)
poly = mpl.patches.Polygon(
    np.transpose(poly.boundary.xy),
    edgecolor="C2",
    facecolor=mpl.colors.to_rgba("C2", alpha=0.1),
)
ax.add_patch(poly)

for node in it.chain.from_iterable(new_casual):
    if node == 7:
        label = "$\\omega_0$"
    else:
        label = "$\\omega_1$"
    label_node(pos[node], label, ax=ax)

# New state.
ax = axes[5]
plot_nodes(pos, include=nodes, ax=ax)
plot_edges(steady, pos, ax=ax)
plot_edges(new_casual, pos, ax=ax, ls="--")

plot_nodes(pos, include=deleted_nodes, ax=ax, **deleted_kwargs)
plot_edges(deleted_steady, pos, ax=ax, **deleted_kwargs)
plot_edges(original_casual, pos, ax=ax, ls="--", **deleted_kwargs)


for ax, title, label in zip(axes, titles, "abcdef"):
    ax.set_title(f"({label}) {title}", fontsize="medium")
    ax.set_aspect("equal")
    # ax.text(0.05, 0.95, f"({label})", va="top", transform=ax.transAxes)
    ax.set_axis_off()

# Only invert once because the y axes are shared. Otherwise, this would lead to no
# inversion because there's an even number of axes.
ax.invert_yaxis()

fig.tight_layout()
fig.savefig("illustration.pdf")
```

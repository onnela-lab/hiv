---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import itertools as it
import json
from matplotlib import pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
from xml.etree import ElementTree

mpl.rcParams["figure.dpi"] = 144
```

```{code-cell} ipython3
# Load the data from the SVG extracted from page 34 of
# https://www.researchgate.net/publication/282683094_The_Social_and_the_Sexual_Networks_in_Contemporary_Demographic_Research
with open("likoma.svg") as fp:
    tree = ElementTree.parse(fp)
```

```{code-cell} ipython3
edges = []
nodes = []
node_attrs = []
aligned = []
for element in tree.iter():
    if element.tag.endswith("path"):
        # Get the style and skip if there is no style or the only attribute is the stroke-width.
        # These paths don't seem to be rendered by Safari or Inkscape.
        if style := element.attrib.get("style"):
            style = dict(x.split(":") for x in element.attrib["style"].split(";"))
        else:
            continue
        if set(style) == {'stroke-width'}:
            continue

        # Load the path and split by "move to" commands which separate different elements of the
        # visualization.
        path = element.attrib["d"]
        for edge in path.split("M"):
            edge = edge.strip()
            if not edge:
                continue
            # Nodes are defined by cubic curves indicated by a C character.
            if any(char in edge for char in "C"):
                # We only care about nodes that are filled because the sex of each node depends on
                # the fill color.
                if (fill := style["fill"]) == "none":
                    continue
                # Remove the "C" character and separate vertices.
                vertices = edge.replace("C", "").split()
                # Drop the "Z" that closes the path if it exists.
                if vertices[-1] == "Z":
                    vertices = vertices[:-1]
                # Parse the vertices (this includes control vertices of the cubic curves). That's
                # not a problem because we're only interested in the centroid.
                vertices = [tuple(map(float, vertex.split(","))) for vertex in vertices]
                # Discard the last vertex which is the same as the first. We don't want to throw off
                # the centroid.
                assert vertices[0] == vertices[-1]
                vertices = vertices[:-1]
                # Compute the mean to get the centroid of the node.
                vertex = np.mean(vertices, axis=0)
                nodes.append(vertex)
                # The scale indicates whether this person was interviewed.
                scale = np.std(vertices, axis=0)[0]
                node_attrs.append({
                    "sex": "female" if fill == "#dedede" else "male",
                    "interviewed": True if scale > 1 else False,
                    "observed": True,
                    "pos": tuple(map(float, vertex)),
                })
                # We're done here.
                continue

            # Horizontal edges are sometimes encoded separately.
            if "H" in edge:
                a, b = map(str.strip, edge.split("H"))
                x1, y1 = map(float, a.split(","))
                x2 = float(b)
                vertices = [(x1, y1), (x2, y1)]
                aligned_ = True
            # Vertical edges are sometimes encoded separately.
            elif "V" in edge:
                a, b = map(str.strip, edge.split("V"))
                x1, y1 = map(float, a.split(","))
                y2 = float(b)
                vertices = [(x1, y1), (x1, y2)]
                aligned_ = True
            # General edges.
            else:
                edge = edge.rstrip("Z")
                vertices = [
                    tuple(map(float, vertex.split(",")))
                    for vertex in edge.split(" ") if vertex
                ]
                aligned_ = False
            # Skip if there are not enough vertices and complain if there are more than two.
            if len(vertices) < 2:
                continue
            if len(vertices) > 2:
                raise ValueError
            # Ignore any potential self loops.
            a, b = vertices
            if a == b:
                continue
            # Store the edge and record whether it was a "special" edge that used the horizontal or
            # vertical line syntax. We need this information to identify problematic edges (see
            # below).
            edges.append(vertices)
            aligned.append(aligned_)

# Cast to arrays and report number of edges and nodes.
edges = np.asarray(edges)
nodes = np.asarray(nodes)
aligned = np.asarray(aligned)
len(nodes), len(edges)
```

```{code-cell} ipython3
# The drawing algorithm is "smart" and combines horizontal or vertical edges that are perfectly
# aligned with one another. This means that some nodes get "skipped" and we end up with missing
# edges. We thus consider all axis-aligned edges and divide them into "reviewed_edges" that aren't a
# problem and "problem_edges" that need to be split up.
reviewed_edges = [
    761, 763, 563, 372, 641, 860, 431, 432, 302, 301, 387, 270, 84, 678, 497, 346, 287,
    286, 1209, 778, 734, 543, 740, 400, 399, 137, 309, 841,
]
problem_edges = [254, 843, 579, 752, 721, 533, 747]
# These are the edges we still need to review.
to_review = aligned & (~np.in1d(np.arange(len(edges)), reviewed_edges + problem_edges))

# Visualize the extracted data.
fig, ax = plt.subplots()
ax.set_aspect("equal")

# Add all the edges.
lines = mpl.collections.LineCollection(edges, alpha=0.5, color="k")
ax.add_collection(lines)
# Add all axis-aligned edges that still require review and corresponding vertices.
lines = mpl.collections.LineCollection(edges[to_review], color="magenta", zorder=99, alpha=0.5,
                                       lw=3)
ax.add_collection(lines)
ax.scatter(*edges[to_review].reshape((-1, 2)).T, marker="x", color="green", zorder=999, alpha=0.5)

# Label the edges so we can manually classify them as "problem" or "reviewed".
for i in np.nonzero(to_review)[0]:
    ax.text(*edges[i].mean(axis=0), str(i), fontsize="x-small", zorder=9999)

# Show all problem edges we have identified.
lines = mpl.collections.LineCollection(edges[problem_edges], color="red", zorder=99, alpha=0.5,
                                       lw=3)
ax.add_collection(lines)

# Plot all the nodes.
ax.scatter(
    *nodes.T, marker=".", zorder=9,
    s=[40 if x["interviewed"] else 20 for x in node_attrs],
    c=np.asarray([x["sex"] == "female" for x in node_attrs]),
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("extracted.pdf")

# For each of the problematic edges, we split it down the middle to ensure we don't skip any nodes.
extra_edges = []
for i in problem_edges:
    a, c = edges[i]
    b = (a + c) / 2
    extra_edges.extend([
        (b, a),
        (b, c),
    ])
extra_edges = np.asarray(extra_edges)

# Now create the filtered edges--our final result.
ok_edges = np.delete(edges, problem_edges, axis=0)
filtered_edges = np.concatenate([ok_edges, extra_edges])
```

```{code-cell} ipython3
# Compute the distance between all pairs of nodes and edges and find the shortest distance for all
# edges. A "good" threshold will separate connected nodes from disconnected nodes.
distances = np.sqrt(np.square(filtered_edges[:, None] - nodes[:, None, :]).sum(axis=-1))
min_distances = distances.min(axis=1)

threshold = 0.1
fig, ax = plt.subplots()
ax.plot(np.sort(min_distances.ravel()), marker=".")
ax.axhline(threshold, color="k", ls=":")
ax.set_yscale("log")
ax.set_ylim(1e-3, 10)
ax.set_xlabel("cumulative edge locations")
ax.set_ylabel("distance to nearest node")
fig.tight_layout()
```

```{code-cell} ipython3
# Now let's construct the graph by finding nodes that are close to the endpoints of edges. Let's
# start by adding all known nodes.
graph = nx.Graph()
graph.add_nodes_from(enumerate(node_attrs))

for i, (pos_u, pos_v) in enumerate(filtered_edges):
    # Find the closest node for the first vertex.
    du = np.square(nodes - pos_u).sum(axis=-1)
    u = np.argmin(du)
    # Create a virtual node if we're missing a close one. We allow a greater tolerance if this edge
    # is an "extra" edge because splitting the problematic edges down the middle might lead to
    # slight misalignments.
    if du[u] > (threshold if i < len(ok_edges) else 20):
        u = graph.number_of_nodes()
        graph.add_node(u, observed=False, pos=tuple(map(float, pos_u)))

    # Same for the second vertex but without the extra tolerance.
    dv = np.square(nodes - pos_v).sum(axis=-1)
    v = np.argmin(dv)
    if dv[v] > threshold:
        v = graph.number_of_nodes()
        graph.add_node(v, observed=False, pos=tuple(map(float, pos_v)))
    graph.add_edge(u, v)


# Post-process the graph by removing unobserved nodes that are sitting right on top of each other.
# They were probably not meant to be there in the first place and should instead connect a group of
# nodes.
unobserved_nodes = [node for node, data in graph.nodes(data=True) if not data["observed"]]
grouped_graph = nx.Graph()
for u, v in it.combinations(unobserved_nodes, 2):
    d = np.square(np.asarray(graph.nodes[u]["pos"]) - graph.nodes[v]["pos"]).sum(axis=-1)
    if d < threshold:
        grouped_graph.add_edge(u, v)

# Join all pairs of neighbors of each group.
for group in nx.connected_components(grouped_graph):
    neighbors = set()
    for node in group:
        neighbors.update(graph.neighbors(node))
    # Connect all pairs of neighbors.
    graph.add_edges_from(it.combinations(neighbors, 2))

# Drop the nodes from the graph.
graph.remove_nodes_from(grouped_graph)
```

```{code-cell} ipython3
# Visualize the graph.
pos = {node: data["pos"] for node, data in graph.nodes(data=True)}
fig, ax = plt.subplots()

nx.draw_networkx_nodes(
    graph, pos,
    node_size=[15 if data.get("interviewed") else 5 for _, data in graph.nodes(data=True)],
    node_color=[
        ("C1" if data["sex"] == "female" else "C0") if data["observed"] else "silver"
        for _, data in graph.nodes(data=True)
    ],
)
nx.draw_networkx_edges(graph, pos, edge_color="gray")
ax.set_axis_off()
```

```{code-cell} ipython3
# Show connected components in different colors.
fig, ax = plt.subplots()
nx.draw_networkx_edges(graph, pos, edge_color="gray")
connected_components = list(nx.connected_components(graph))
print(f"found {len(connected_components)} connected components")
for component, (c, m) in zip(connected_components, it.product(range(10), '*svo.')):
    c = f"C{c}"
    nx.draw_networkx_nodes(graph, pos, component, node_color=c, node_shape=m,
                           node_size=10, alpha=0.5)

# Little crosses for unobserved nodes.
unobserved = [node for node, data in graph.nodes(data=True) if not data["observed"]]
nx.draw_networkx_nodes(graph, pos, unobserved,
                       node_shape="x", node_size=10, node_color="k")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("connected_components.pdf")
```

```{code-cell} ipython3
# Save the result as JSON.
def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise ValueError

with open("likoma.json", "w") as fp:
    json.dump({
        "nodes": list(graph.nodes(data=True)),
        "edges": list(graph.edges),
    }, fp, default=default, indent=4, sort_keys=True)
```

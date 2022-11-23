import logging
import networkx as nx
import numpy as np
import pytest
from hiv import stockholm
from hiv.util import assert_graphs_equal


LOGGER = logging.getLogger(__name__)


@pytest.fixture(params=[
    {"n": 50, "w0": 0.2, "w1": 0.1, "mu": 0.3, "rho": 0.4, "sigma": 0.5},
])
def params(request: pytest.FixtureRequest) -> dict[str, float]:
    return request.param | {"verify": True}


def test_add_nodes_from_update() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5), a="a")
    updated = {2, 3, 7}
    graph.add_nodes_from(updated, a="b")
    for node in updated:
        assert graph.nodes[node]["a"] == "b"
    for node in set(graph) - updated:
        assert graph.nodes[node]["a"] == "a"

    assert graph.number_of_nodes() == 6


def test_multi_step(params: dict[str, float]) -> None:
    seed = 0
    num_steps = 2
    np.random.seed(seed)
    LOGGER.info("starting multi-step simulation...")
    graph1, statistics1 = stockholm.simulate(**params, num_steps=num_steps)
    np.random.seed(seed)
    LOGGER.info("starting sequential simulation...")
    graph2 = nx.Graph()
    for step in range(num_steps):
        _, statistics2 = stockholm.simulate(**params, num_steps=1, step=step, graph=graph2)

    # Verify we have the same graph and statistics.
    assert_graphs_equal(graph1, graph2)
    assert set(statistics1) == set(statistics2)
    for key in statistics1:
        np.testing.assert_allclose(statistics1[key][-1], statistics2[key][-1])

    # Verify expected values.
    for key, graph, statistics in [("multi", graph1, statistics1), ("seq", graph2, statistics2)]:
        assert stockholm.evaluate_num_edges(graph, True) == statistics["num_casual_edges"][-1], \
            f"number of casual relationships is wrong for {key}"
        assert stockholm.evaluate_num_edges(graph, False) == statistics["num_steady_edges"][-1], \
            f"number of steady relationships is wrong for {key}"


def test_many_steps(params: dict[str, float]) -> None:
    graph, _ = stockholm.simulate(**params, num_steps=1000)
    for node, data in graph.nodes(data=True):
        edges = graph.edges(node, data=True)
        if data["is_single"]:
            assert not edges or all(edge_data["is_casual"] for *_, edge_data in edges)
        else:
            assert edges and sum(1 for *_, edge_data in edges if not edge_data["is_casual"]) == 1
        if data["has_casual"]:
            assert edges and sum(1 for *_, edge_data in edges if edge_data["is_casual"]) == 1
        else:
            assert not edges or not any (edge_data["is_casual"] for *_, edge_data in edges)

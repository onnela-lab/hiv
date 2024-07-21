import logging
import networkx as nx
import pytest
from hiv import stockholm


LOGGER = logging.getLogger(__name__)


@pytest.fixture(
    params=[
        {"n": 50, "w0": 0.2, "w1": 0.1, "mu": 0.3, "rho": 0.4, "sigma": 0.5},
    ]
)
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


def test_many_steps(params: dict[str, float]) -> None:
    graph = nx.empty_graph()
    for _ in range(1000):
        graph = stockholm.step(graph, **params)
    for node, data in graph.nodes(data=True):
        edges = graph.edges(node, data=True)
        if data["is_single"]:
            assert not edges or all(edge_data["is_casual"] for *_, edge_data in edges)
        else:
            assert (
                edges
                and sum(1 for *_, edge_data in edges if not edge_data["is_casual"]) == 1
            )
        if data["has_casual"]:
            assert (
                edges
                and sum(1 for *_, edge_data in edges if edge_data["is_casual"]) == 1
            )
        else:
            assert not edges or not any(
                edge_data["is_casual"] for *_, edge_data in edges
            )

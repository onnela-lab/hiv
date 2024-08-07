import networkx as nx
from hiv.simulator import (
    estimate_paired_fraction,
    Interval,
    number_of_nodes,
    UniversalSimulator,
)
import pytest


@pytest.mark.parametrize(
    "simulator",
    [
        UniversalSimulator(
            n=50, rho=0.3, sigma=0.5, xi=0.9, omega0=0.4, omega1=0.2, mu=0.1
        ),
        UniversalSimulator(
            n=1, rho=0.3, sigma=0.5, xi=0.9, omega0=0.4, omega1=0.2, mu=0.1
        ),
    ],
)
def test_simulator(simulator: UniversalSimulator) -> None:
    graph0 = simulator.init()
    simulator.step(graph0)

    graph1 = graph0.copy()
    simulator.step(graph1)
    simulator.evaluate_summaries(graph0, graph1)

    simulator.run(graph1, 100, validate=True)
    simulator.evaluate_summaries(graph0, graph1)


def test_invalid_param() -> None:
    with pytest.raises(ValueError, match="`n` does not satisfy constraint"):
        UniversalSimulator(
            n=-1, rho=0.3, sigma=0.5, xi=0.9, omega0=0.4, omega1=0.2, mu=0.1
        )


def test_invalid_type() -> None:
    constraint = Interval(0, None, type=int)
    assert constraint.is_valid(17)
    assert not constraint.is_valid(17.0)
    assert not constraint.is_valid(-1)
    assert "type=<class 'int'>" in repr(constraint)


def test_number_of_nodes() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(
        [(0, {"foo": "bar"}), (1, {"foo": "bazz"}), (2, {"foo": "bazz"})]
    )
    assert number_of_nodes(graph) == 3
    assert number_of_nodes(graph, foo="bazz") == 2
    assert number_of_nodes(graph, lambda x: x["foo"] == "bar") == 1


def test_estimate_paired_fraction() -> None:
    assert estimate_paired_fraction(0.2, 0, 0) == 1
    assert estimate_paired_fraction(0, 0.5, 0.7) == 0

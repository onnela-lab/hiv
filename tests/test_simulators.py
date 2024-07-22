from hiv import simulators
import pytest


@pytest.mark.parametrize(
    "simulator",
    [
        simulators.KretzschmarMorris(50, 0.7, 0.5, 0.3),
        simulators.Stockholm(50, 0.3, 0.5, 0.4, 0.2, 0.1),
    ],
)
def test_simulator(simulator: simulators.Simulator) -> None:
    graph = simulator.init()
    simulator.run(graph, 100)


def test_invalid_param() -> None:
    with pytest.raises(ValueError, match="`n` does not satisfy constraint"):
        simulators.KretzschmarMorris(0.3, 0.3, 0.3, 0.3)

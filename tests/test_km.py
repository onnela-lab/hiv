from hiv import km


def test_simulate_km() -> None:
    graph, stats = km.simulate(10, 0.3, 0.7, 0.5, 500)
    assert graph.number_of_nodes() == 10
    for value in stats.values():
        assert value.shape == (500,)

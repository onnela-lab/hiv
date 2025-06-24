from hiv.simulator import estimate_paired_fraction, Interval, UniversalSimulator
import numpy as np
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
@pytest.mark.parametrize("sample_size_frac", [None, 0.1, 0.5, 1.0])
def test_simulator(
    simulator: UniversalSimulator, sample_size_frac: float | None
) -> None:
    graph0 = simulator.init()
    simulator.step(graph0)

    graph1 = graph0.copy()
    simulator.step(graph1)
    simulator.evaluate_summaries(graph0, graph1)

    simulator.run(graph1, 100, validate=True)

    if sample_size_frac is None or graph0.nodes.size == 0:
        sample0 = None
    else:
        sample_size = max(1, int(sample_size_frac * graph0.nodes.size))
        sample0 = np.random.choice(graph0.nodes, sample_size, replace=False)
    simulator.evaluate_summaries(graph0, graph1, sample0)


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


def test_estimate_paired_fraction() -> None:
    np.testing.assert_allclose(estimate_paired_fraction(0.2, 0, 0), 1)
    np.testing.assert_allclose(estimate_paired_fraction(0, 0.5, 0.7), 0)
    # Non-zero lag should have larger paired fraction ...
    np.testing.assert_array_less(
        estimate_paired_fraction(0.1, 0.2, 0.3, 10),
        estimate_paired_fraction(0.1, 0.2, 0.3, 20),
    )
    # ... unless the migration rate is zero.
    np.testing.assert_allclose(
        estimate_paired_fraction(0.1, 0, 0.3, 10),
        estimate_paired_fraction(0.1, 0, 0.3, 20),
    )
    # Zero lag is the same as no lag.
    np.testing.assert_allclose(
        estimate_paired_fraction(0.1, 0.2, 0.3, lag=None),
        estimate_paired_fraction(0.1, 0.2, 0.3, lag=0),
    )

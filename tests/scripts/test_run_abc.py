from hiv.scripts import generate_data, run_abc
import pathlib
import pytest


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--adjust"],
        ["--standardize=local"],
        ["--standardize=global"],
        ["--save-samples"],
        ["--max-lag=6"],
    ],
)
def test_run_abc(argv: list[str], tmp_path: pathlib.Path) -> None:
    # Generate synthetic data.
    num_samples = 10
    num_lags = 7
    train_path = tmp_path / "train/output.pkl"
    test_path = tmp_path / "test/output.pkl"
    generate_data.__main__(["--param=n=17", num_samples, num_lags, train_path])
    generate_data.__main__(["--param=n=17", num_samples - 3, num_lags, test_path])

    # Run the inference.
    result_path = tmp_path / "result.pkl"
    run_abc.__main__(
        argv + ["--frac=0.5", train_path.parent, test_path.parent, result_path]
    )
    assert result_path.is_file()

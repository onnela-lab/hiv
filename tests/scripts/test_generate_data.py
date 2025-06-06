from hiv.scripts import generate_data
import networkx as nx
import pathlib
import pickle
import pytest


@pytest.mark.parametrize(
    "argv",
    [
        ("--param=n=23.", "--param=omega0=beta:2,2", "--preset=hansson2019"),
        ("--param=n=17", "--param=rho=beta:2,2", "--preset=kretzschmar1996"),
        ("--param=n=17", "--param=rho=beta:2,2", "--preset=kretzschmar1998"),
        ("--param=n=17", "--param=rho=beta:2,2", "--preset=leng2018", "--seed=3"),
        (
            "--param=n=17",
            "--param=mu=0.95",
            "--param=rho=beta:2,2",
            "--preset=leng2018",
            "--seed=3",
        ),
    ],
)
@pytest.mark.parametrize("save_graphs", [False, True])
@pytest.mark.parametrize("sample_size", [10, None])
def test_generate_data(
    argv: tuple, save_graphs: bool, tmp_path: pathlib.Path, sample_size: int | None
) -> None:
    num_samples = 27
    num_lags = 5
    output = tmp_path / "output.pkl"
    if sample_size:
        argv += (f"--sample-size={sample_size}",)
    if save_graphs:
        argv += ("--save-graphs",)
    argv += (num_samples, num_lags, output)
    generate_data.__main__(list(map(str, argv)))

    with open(output, "rb") as fp:
        result = pickle.load(fp)

    if save_graphs:
        graph_sequences: list[list[nx.Graph]] = result["graph_sequences"]
        assert len(graph_sequences) == num_samples
    else:
        assert "graph_sequences" not in result

    for values in result["params"].values():
        assert values.shape == (num_samples,)

    for key, values in result["summaries"].items():
        if not key.startswith("_"):
            assert values.shape == (num_samples, num_lags)
            if key.startswith("frac"):
                assert (values >= 0).all(), "probability is negative"
                assert (values <= 1).all(), "probability is larger than one"

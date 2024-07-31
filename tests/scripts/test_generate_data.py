from hiv.scripts import generate_data
from hiv.simulator import number_of_nodes
import networkx as nx
import numpy as np
import pathlib
import pickle
import pytest


@pytest.mark.parametrize(
    "argv",
    [
        ["--param=n=23.", "--param=omega0=beta:2,2", "--preset=hansson2019"],
        ["--param=n=17", "--param=rho=beta:2,2", "--preset=kretzschmar1996"],
        ["--param=n=17", "--param=rho=beta:2,2", "--preset=kretzschmar1998"],
        ["--param=n=17", "--param=rho=beta:2,2", "--preset=leng2018"],
    ],
)
@pytest.mark.parametrize("save_graphs", [False, True])
def test_generate_data(argv: list, save_graphs: bool, tmp_path: pathlib.Path) -> None:
    num_samples = 7
    num_lags = 5
    output = tmp_path / "output.pkl"
    simulator = argv[-1]
    argv = argv + [num_samples, num_lags, output]
    if save_graphs:
        argv.append("--save_graphs")
    generate_data.__main__(list(map(str, argv)))

    with open(output, "rb") as fp:
        result = pickle.load(fp)

    if save_graphs:
        graph_sequences: list[list[nx.Graph]] = result["graph_sequences"]
        assert len(graph_sequences) == num_samples
        for i, graph_sequence in enumerate(graph_sequences):
            assert len(graph_sequence) == num_lags
            # Check that summaries are consistent with the graph.
            if simulator == "stockholm":
                num_paired0 = number_of_nodes(graph_sequence[0], is_single=False)
                num0 = graph_sequence[0].number_of_nodes()
                for j, graph in enumerate(graph_sequence):
                    assert graph.number_of_nodes() == number_of_nodes(
                        graph, is_single=True
                    ) + number_of_nodes(graph, is_single=False)
                    frac_paired = (
                        number_of_nodes(graph, is_single=False) + num_paired0
                    ) / (graph.number_of_nodes() + num0)
                    np.testing.assert_allclose(
                        result["summaries"]["frac_paired"][i, j], frac_paired
                    )
    else:
        assert "graph_sequences" not in result

    for values in result["params"].values():
        assert values.shape == (num_samples,)

    for values in result["summaries"].values():
        assert values.shape == (num_samples, num_lags)

from hiv.scripts import generate_data
from hiv.stockholm import evaluate_num_nodes
import networkx as nx
import numpy as np
import pytest
from unittest.mock import patch


@pytest.mark.parametrize("save_graphs", [False, True])
def test_generate_data(save_graphs: bool) -> None:
    num_samples = 7
    n = 23
    num_lags = 5
    args = [num_samples, n, num_lags, "/not/a/file"]
    if save_graphs:
        args.append("--save_graphs")
    with patch("builtins.open"), patch("pickle.dump") as dump:
        generate_data.__main__(list(map(str, args)))

    dump.assert_called_once()
    (result, _), _ = dump.call_args

    if save_graphs:
        graph_sequences: list[list[nx.Graph]] = result["graph_sequences"]
        assert len(graph_sequences) == num_samples
        for i, graph_sequence in enumerate(graph_sequences):
            assert len(graph_sequence) == num_lags
            # Check that summaries are consistent with the graph.
            num_paired0 = evaluate_num_nodes(graph_sequence[0], False)
            num0 = graph_sequence[0].number_of_nodes()
            for j, graph in enumerate(graph_sequence):
                assert graph.number_of_nodes() == evaluate_num_nodes(
                    graph, True
                ) + evaluate_num_nodes(graph, False)
                frac_paired = (evaluate_num_nodes(graph, False) + num_paired0) / (
                    graph.number_of_nodes() + num0
                )
                np.testing.assert_allclose(
                    result["summaries"]["frac_paired"][i, j], frac_paired
                )
    else:
        assert "graph_sequences" not in result

    for values in result["params"].values():
        assert values.shape == (num_samples,)

    for values in result["summaries"].values():
        assert values.shape == (num_samples, num_lags)

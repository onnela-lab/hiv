from hiv.scripts import generate_data
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
        graph_sequences = result["graph_sequences"]
        assert len(graph_sequences) == num_samples
        for graph_sequence in graph_sequences:
            assert len(graph_sequence) == num_lags
    else:
        assert "graph_sequences" not in result

    for values in result["params"].values():
        assert values.shape == (num_samples,)

    for values in result["summaries"].values():
        assert values.shape == (num_samples, num_lags)

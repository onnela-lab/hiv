from hiv.scripts.create_hansson_data import __main__
from hiv.scripts.run_abc import load_batches
from pathlib import Path


def test_create_hansson_data(tmp_path: Path) -> None:
    __main__([str(tmp_path / "test.pkl")])
    summaries, params = load_batches(tmp_path)
    assert len(summaries) == 5
    assert len(params) == 6

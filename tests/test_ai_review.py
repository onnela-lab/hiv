import asyncio
from hiv.scripts.ai_review import __main__
from pathlib import Path


def test_ai_review(tmp_path: Path) -> None:
    # We pass in an arbitrary file because the test model won't process it anyway.
    path = tmp_path / "test.pdf"
    path.write_text("empty")
    asyncio.run(__main__(["--model=test", str(path)]))

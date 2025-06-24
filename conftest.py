import os
import numpy as np
import pytest

os.environ["CI"] = "True"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a fixed random seed for reproducible tests."""
    np.random.seed(42)

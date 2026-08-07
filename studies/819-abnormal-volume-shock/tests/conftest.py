"""Shared fixtures — deterministic synthetic abnormal-volume panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volume_shock import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted attention->drift relation: high abnormal volume, high forward return."""
    return data.synthetic_panel(edge=0.0020, seed=819, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — abnormal volume varies across names but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=819, n_assets=40, n_days=1500)

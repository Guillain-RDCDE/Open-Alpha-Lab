"""Shared fixtures — deterministic synthetic vol-trend panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_trend import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted rising-vol->de-rate relation: rising vol de-rates, falling vol re-rates."""
    return data.synthetic_panel(edge=0.0015, seed=817, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — the vol trend varies across names but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=817, n_assets=40, n_days=1500)

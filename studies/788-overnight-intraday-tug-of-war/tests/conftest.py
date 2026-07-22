"""Shared fixtures — deterministic synthetic tug-of-war panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from overnight_intraday_tug import data  # noqa: E402


@pytest.fixture(scope="session")
def tug_world():
    """A planted Lou-Polk-Skouras tug: overnight persists, intraday reverses."""
    return data.synthetic_panel(tug=0.0020, seed=788, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — trailing overnight return carries no information about either leg."""
    return data.synthetic_panel(tug=0.0, seed=788, n_assets=40, n_days=1500)

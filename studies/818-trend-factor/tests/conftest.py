"""Shared fixtures — deterministic synthetic trend panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trend_factor import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted trend->return relation: persistent trends both move the price and predict
    the next return, so the fitted trend factor should light up long-high / short-low."""
    return data.synthetic_panel(edge=0.0015, seed=818, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — prices are random walks, the moving-average signals carry no forward
    information, and the fitted trend factor must find nothing."""
    return data.synthetic_panel(edge=0.0, seed=818, n_assets=40, n_days=1500)

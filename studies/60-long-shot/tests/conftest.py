"""Shared fixtures — deterministic synthetic commodity-skewness panels (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from long_shot import data


@pytest.fixture(scope="session")
def skew_world():
    """High-skew commodities genuinely underperform — the textbook trade should pay."""
    return data.synthetic_panel(n_assets=18, skew_premium=0.0030, seed=60)


@pytest.fixture(scope="session")
def null_world():
    """The null — skewness carries no information."""
    return data.synthetic_panel(n_assets=18, skew_premium=0.0, seed=60)

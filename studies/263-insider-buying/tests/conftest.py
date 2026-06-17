"""Shared fixtures -- deterministic synthetic (ratio, forward-return) panels with a
*known* amount of contrarian predictive power, so tests never touch the network and
the only thing the regression can detect is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insider_buying import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No planted effect (beta=0). The ratio carries no forecasting information."""
    df, truth = data.synthetic_panel(beta=0.0, seed=263)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted contrarian beta. The regression should detect it (t > 2)."""
    df, truth = data.synthetic_panel(beta=0.06, seed=263)
    return df, truth

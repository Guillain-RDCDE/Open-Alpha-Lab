"""Shared fixtures -- deterministic synthetic (IPO count, forward-return) panels with a
*known* froth slope, so tests never touch the network and the only thing the predictive
regression can recover is baked in (beta < 0) or deliberately absent (beta = 0)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_volume import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No froth slope (beta=0). The predictive regression should read ~zero."""
    df, truth = data.synthetic_annual(n_years=60, beta=0.0, seed=265)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted negative froth slope (beta=-0.06). High IPO volume -> low forward return."""
    df, truth = data.synthetic_annual(n_years=120, beta=-0.06, seed=265)
    return df, truth

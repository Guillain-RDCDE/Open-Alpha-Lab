"""Shared fixtures -- deterministic synthetic cross-sections with a *known*
short-interest -> forward-return slope, so tests never touch the network and the
only thing the quantile sort can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_interest import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No SI->return slope (signal = noise). Quantile spread should be ~0."""
    df, truth = data.synthetic_panel(n_firms=200, si_beta=0.0, seed=262)
    return df, truth


@pytest.fixture
def bleed_panel():
    """Planted negative slope: high-SI stocks bleed. Spread should be < 0, t < -2."""
    df, truth = data.synthetic_panel(n_firms=200, si_beta=-0.04, seed=262)
    return df, truth


@pytest.fixture
def bounce_panel():
    """Planted positive slope: high-SI stocks bounce. Spread should be > 0, t > 2."""
    df, truth = data.synthetic_panel(n_firms=200, si_beta=0.04, seed=262)
    return df, truth

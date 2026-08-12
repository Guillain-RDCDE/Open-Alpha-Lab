"""Shared fixtures — deterministic synthetic butterfly tapes with a *known* amount of
curvature predictive power, so tests never touch the network and the only thing the
regression / sort can harvest (the planted twist edge) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_twist import data  # noqa: E402


@pytest.fixture(scope="session")
def null_tape():
    """A tape where the butterfly carries no forward information (fly_signal=0)."""
    df, truth = data.synthetic_daily(n_days=3000, fly_signal=0.0, seed=864)
    return df, truth


@pytest.fixture(scope="session")
def signal_tape():
    """A tape with a planted butterfly -> forward-IEF-return link (the positive control)."""
    df, truth = data.synthetic_daily(n_days=3000, fly_signal=0.02, seed=864)
    return df, truth

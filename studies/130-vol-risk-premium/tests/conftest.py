"""Shared fixtures — deterministic synthetic VIX/SPY daily tapes with a *known*
amount of VRP predictive power, so tests never touch the network and the
effect under study can be dialled on or off."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_risk_premium import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where VRP carries no predictive power (vrp_signal=0, the null)."""
    df, truth = data.synthetic_daily(n_days=2000, vrp_signal=0.0, seed=130)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with planted VRP signal (vrp_signal=0.02) — the engine should find it."""
    df, truth = data.synthetic_daily(n_days=2000, vrp_signal=0.02, seed=130)
    return df, truth

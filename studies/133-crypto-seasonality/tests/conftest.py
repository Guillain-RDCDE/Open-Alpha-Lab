"""Shared fixtures — deterministic synthetic daily BTC tapes with a *known* monthly
boost, so tests never touch the network and the only thing the seasonal rule can
harvest is planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_seasonality import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk tape — no calendar month has planted signal."""
    bars, truth = data.synthetic_daily(n_years=11, monthly_boost=0.0, seed=133)
    return bars, truth


@pytest.fixture
def uptober_tape():
    """A tape with a strong October boost (monthly_boost=0.50) — the engine must find it."""
    bars, truth = data.synthetic_daily(
        n_years=11, monthly_boost=0.50, boost_month=10, seed=133
    )
    return bars, truth

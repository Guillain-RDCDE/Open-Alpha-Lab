"""Shared fixtures — deterministic synthetic (btc, mstr) pairs with a *known* MSTR
construction, so tests never touch the network and the only thing that could make MSTR
beat levered BTC (a planted alpha) is either baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_treasury import data  # noqa: E402


@pytest.fixture
def null_pair():
    """Pure levered beta, no alpha — 'hold MSTR not BTC' must be a wash here."""
    frame, truth = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0,
                                       premium_vol=0.0, seed=324)
    return frame, truth


@pytest.fixture
def alpha_pair():
    """Levered beta + a genuine planted daily alpha — the harness must detect it."""
    frame, truth = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0010,
                                       premium_vol=0.0, seed=324)
    return frame, truth


@pytest.fixture
def premium_pair():
    """Levered beta, no alpha, but a swinging NAV premium — return-free extra variance."""
    frame, truth = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0,
                                       premium_vol=0.30, seed=324)
    return frame, truth

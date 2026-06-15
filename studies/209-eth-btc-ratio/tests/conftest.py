"""Shared fixtures — deterministic synthetic ETH/BTC tapes with a *known* amount of
ratio momentum, so tests never touch the network and the only thing the rotation
signal can harvest (ratio trend persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from eth_btc_ratio import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale ETH/BTC ratio (momentum=0) — the rotation strategy is a coin here."""
    bars, truth = data.synthetic_daily(n_days=500, ratio_momentum=0.0, seed=209)
    return bars, truth


@pytest.fixture
def trending_tape():
    """A tape with real ratio persistence (ratio_momentum=0.30) — the rotation should win here."""
    bars, truth = data.synthetic_daily(n_days=500, ratio_momentum=0.30, seed=209)
    return bars, truth

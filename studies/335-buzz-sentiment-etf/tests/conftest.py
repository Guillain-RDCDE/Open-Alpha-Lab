"""Shared fixtures — deterministic synthetic BUZZ/SPY pairs with a *known* amount
of alpha, so tests never touch the network and the only thing the CAPM-alpha test
can certify (real skill on top of beta and fees) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buzz_sentiment_etf import data  # noqa: E402


@pytest.fixture
def null_pair():
    """A dressed-up-beta null (alpha=0): BUZZ is SPY scaled by beta, plus noise,
    minus a fee. The alpha test must read 'no alpha' here."""
    levels, truth = data.synthetic_pair(n_days=3000, alpha_bps=0.0, seed=335)
    return levels, truth


@pytest.fixture
def alpha_pair():
    """A planted positive alpha (4 bps/day ≈ +10%/yr) — the engine must recover it."""
    levels, truth = data.synthetic_pair(n_days=3000, alpha_bps=4.0, seed=335)
    return levels, truth

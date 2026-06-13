"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing the band-pierce
can harvest (short-term mean reversion) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bollinger_reversion import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (reversion 0) — band entries must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, reversion=0.0, seed=104)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with real mean-reversion (reversion 0.20) — band entries should outperform here."""
    bars, truth = data.synthetic_daily(n_days=800, reversion=0.20, seed=104)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with momentum (reversion -0.05) — band entries are on the *wrong* side."""
    bars, truth = data.synthetic_daily(n_days=500, reversion=-0.05, seed=104)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
return serial dependence, so tests never touch the network and the only thing
streak-conditioning can harvest (persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hot_hand import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (autocorr 0) — streaks must carry no information here."""
    bars, truth = data.synthetic_daily(n_days=1000, autocorr=0.0, seed=176)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real daily persistence (autocorr 0.15) — hot-hand should win."""
    bars, truth = data.synthetic_daily(n_days=1500, autocorr=0.15, seed=176)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with daily mean reversion (autocorr -0.15) — gambler's fallacy right."""
    bars, truth = data.synthetic_daily(n_days=1500, autocorr=-0.15, seed=176)
    return bars, truth

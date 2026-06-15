"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion or momentum, so tests never touch the network and the only structure the
Fisher/Trigger crossover can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fisher_transform import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (mean_rev=0) — the Fisher signal must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=183)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with planted mean-reversion (mean_rev=0.30) — the signal should find a hint here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.30, seed=183)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with planted momentum (mean_rev=-0.20) — the signal is on the wrong side here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=-0.20, seed=183)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion or momentum, so tests never touch the network and the structures CMO
is designed to harvest (mean-reversion for OB/OS; momentum for zero-cross) can be
planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from chande_momentum import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (mean_rev=0, momentum=0) — both CMO framings must behave
    like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, momentum=0.0, seed=185)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with planted mean-reversion — the OB/OS framing should find a signal here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.30, momentum=0.0, seed=185)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with planted positive momentum — the zero-cross framing should find a
    signal here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, momentum=0.30, seed=185)
    return bars, truth

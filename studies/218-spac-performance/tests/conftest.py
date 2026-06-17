"""Shared fixtures — deterministic synthetic daily price frames with a *known* planted
alpha, so tests never touch the network and the CAPM recovery is provably correct."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spac_performance import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A SPAC basket with zero planted alpha — the null: SPACs match the market after beta."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=0.0, beta=1.5, seed=218)
    return prices, truth


@pytest.fixture
def positive_tape():
    """A SPAC basket with planted +5%/yr alpha — the engine must detect this."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=0.05, beta=1.5, seed=218)
    return prices, truth


@pytest.fixture
def negative_tape():
    """A SPAC basket with planted -30%/yr alpha — mirrors the real SPAK result."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=-0.30, beta=1.5, seed=218)
    return prices, truth

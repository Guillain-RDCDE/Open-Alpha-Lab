"""Shared fixtures — deterministic synthetic daily price frames with a *known* planted
alpha, so tests never touch the network and the CAPM recovery is provably correct."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ai_powered_etf import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A fund with zero planted alpha (beta=1.05) — the null: AI adds nothing."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=0.0, beta=1.05, seed=139)
    return prices, truth


@pytest.fixture
def positive_tape():
    """A fund with planted +5%/yr alpha — the engine must detect this."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=0.05, beta=1.05, seed=139)
    return prices, truth


@pytest.fixture
def negative_tape():
    """A fund with planted -4.4%/yr alpha — mirrors the real AIEQ result."""
    prices, truth = data.synthetic_daily(n_days=600, alpha_ann=-0.044, beta=1.05, seed=139)
    return prices, truth

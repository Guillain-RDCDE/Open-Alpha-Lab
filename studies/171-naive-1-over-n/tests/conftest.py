"""Shared fixtures — deterministic synthetic return matrices with a *known* structure,
so tests never touch the network and the only thing the optimiser can find (a planted
return differential) is present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from naive_1_over_n import data  # noqa: E402


@pytest.fixture
def null_returns():
    """All assets have the same true expected return — 1/N is exactly optimal here.
    1260 periods ~ 5 years; with 24-month lookback we get ~3 out-of-sample years."""
    df, truth = data.synthetic_returns(n_periods=1260, signal_strength=0.0, seed=171)
    return df, truth


@pytest.fixture
def signal_returns():
    """Assets have a genuine return gradient — an optimiser with a perfect estimate wins here."""
    df, truth = data.synthetic_returns(n_periods=1260, signal_strength=0.005, seed=171)
    return df, truth

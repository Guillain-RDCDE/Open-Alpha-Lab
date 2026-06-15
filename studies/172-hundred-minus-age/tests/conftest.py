"""Shared fixtures — deterministic synthetic lifecycle tapes with a *known* equity
premium, so tests never touch the network and the engine's core logic can be verified
against planted parameters."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hundred_minus_age import data  # noqa: E402


@pytest.fixture
def synth_no_premium():
    """Synthetic tape where bonds and equities have the same expected return.

    In this null world the glidepath gives up nothing in exchange for lower vol —
    the 'insurance is free' control case.  We generate 42 years so that the
    pct_change() step leaves a full 40*12 = 480 monthly return rows available
    to the 40-year lifecycle simulator.
    """
    prices, truth = data.synthetic_lifecycle(
        n_years=42, equity_premium=0.0, seed=172
    )
    # Convert price levels to monthly returns; first row is NaN after the diff.
    returns = prices.pct_change().dropna()
    return returns, truth


@pytest.fixture
def synth_with_premium():
    """Synthetic tape where equities earn a 4% annual real premium over bonds.

    Here the glidepath genuinely forfeits return in exchange for lower volatility
    — the central trade-off the study is built to measure.  42 years generated
    so that 480 monthly return rows are available after the pct_change() step.
    """
    prices, truth = data.synthetic_lifecycle(
        n_years=42, equity_premium=0.04, seed=172
    )
    returns = prices.pct_change().dropna()
    return returns, truth

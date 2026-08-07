"""Shared fixtures — deterministic synthetic worlds (a true-zero-Sharpe null the demo lives on,
and a genuinely-skilful positive control). No network, no real data; fixed seed 834."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from min_backtest_length import data  # noqa: E402


@pytest.fixture(scope="session")
def null_returns():
    """A worthless world: true annualised Sharpe 0 — any in-sample edge is luck."""
    ret, truth = data.synthetic_returns(sr_ann=0.0, n_years=40.0, seed=834)
    return ret, truth


@pytest.fixture(scope="session")
def skilled_returns():
    """The positive control: a genuine annualised Sharpe of 1.0."""
    ret, truth = data.synthetic_returns(sr_ann=1.0, n_years=40.0, seed=834)
    return ret, truth


@pytest.fixture(scope="session")
def fat_left_returns():
    """A negatively-skewed, fat-left-tailed world (skew -2, kurtosis 9) at Sharpe 1.0."""
    ret, truth = data.synthetic_returns(
        sr_ann=1.0, n_years=200.0, dist="fat_left", skew_target=-2.0, seed=834
    )
    return ret, truth

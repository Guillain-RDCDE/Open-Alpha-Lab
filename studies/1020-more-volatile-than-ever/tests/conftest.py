"""Shared fixtures for Study 1020 — deterministic, offline, no network.

The synthetic world is a stochastic-volatility process whose log volatility is a persistent
AR(1). The planted world adds an upward trend that triples volatility across the sample; the
null is the same process with the knob at zero. A persistent stationary volatility series is
precisely the null on which naive trend tests misfire, so a detector that passes on the plant
and *also* on the null is not testing anything — both are exercised below.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volhistory import data  # noqa: E402

N_YEARS = 90


@pytest.fixture
def planted():
    """Monthly returns with a planted upward trend in log volatility (signal_strength=1)."""
    return data.synthetic_returns(n_years=N_YEARS, freq="M", signal_strength=1.0, seed=1020)


@pytest.fixture
def null_world():
    """The matched null: the same persistent stochastic volatility, no trend."""
    return data.synthetic_returns(n_years=N_YEARS, freq="M", signal_strength=0.0, seed=1020)


@pytest.fixture
def daily_null():
    """Thirty years of daily returns, constant-in-expectation volatility, with drift."""
    return data.synthetic_returns(n_years=30, freq="D", signal_strength=0.0, seed=1020)

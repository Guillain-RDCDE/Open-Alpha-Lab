"""Shared fixtures — deterministic synthetic tapes, so tests never touch the network and the
quantities the diagnostics hunt are baked in: a **clustered** tape with a persistent vol regime
(forecastable variance + a real Sharpe gain to recover) and a **flat-vol** null (no regime, so the
overlay must add nothing)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from storm_shy import data, vol


@pytest.fixture
def regime():
    """~5040 bars (20y), persistent calm/storm vol regime (the clustered tape the overlay reads)."""
    close, truth = data.synthetic_prices(n_bars=5040, seed=16)
    return close, truth


@pytest.fixture
def flat():
    """~5040 bars, a single constant vol — the null: no clustering, nothing to forecast."""
    close, truth = data.synthetic_prices(n_bars=5040, sigma_lo=0.011, sigma_hi=0.011, seed=16)
    return close, truth


@pytest.fixture
def regime_returns(regime):
    return vol.to_returns(regime[0])


@pytest.fixture
def flat_returns(flat):
    return vol.to_returns(flat[0])

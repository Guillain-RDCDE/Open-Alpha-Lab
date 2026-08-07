"""Shared fixtures — deterministic synthetic worlds (no network, no real data).

Three worlds, all seeded: two INDEPENDENT random walks (the pitfall the demo dissects),
two INDEPENDENT stationary series (the specificity control that must be correctly sized),
and a genuinely COINTEGRATED pair (the positive control the machinery must detect)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spurious_regression import data  # noqa: E402


@pytest.fixture(scope="session")
def walks():
    """Two independent random walks — where the level-OLS t-stat over-rejects."""
    return data.independent_walks(3000, n_obs=250, seed=835)


@pytest.fixture(scope="session")
def stationary():
    """Two independent stationary (white-noise) series — level OLS must be correctly sized."""
    return data.stationary_pairs(3000, n_obs=250, phi=0.0, seed=835)


@pytest.fixture(scope="session")
def cointegrated():
    """A genuinely cointegrated pair (common trend) — the positive control."""
    return data.cointegrated_pairs(200, n_obs=250, beta=1.0, noise_sd=1.0, seed=835)

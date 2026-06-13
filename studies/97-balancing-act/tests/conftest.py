"""Shared fixtures for Study 97 (Balancing-Act) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from balancing_act import data  # noqa: E402


@pytest.fixture
def diversifying():
    """Two negatively-correlated legs (corr -0.4) — the 60/40 blend must beat the best leg."""
    return data.synthetic_two_asset(n_days=6000, corr=-0.40, seed=97)


@pytest.fixture
def redundant():
    """Two positively-correlated legs (corr +0.85) — blending earns no diversification."""
    return data.synthetic_two_asset(n_days=6000, corr=+0.85, seed=97)

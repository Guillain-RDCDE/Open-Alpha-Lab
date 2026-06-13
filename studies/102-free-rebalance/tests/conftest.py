"""Shared fixtures for Study 102 (Free-Rebalance) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from free_rebalance import data  # noqa: E402


@pytest.fixture
def mean_revert():
    """Two equal-drift, mean-reverting assets — the rebalancing bonus must be POSITIVE."""
    return data.synthetic_meanrevert(n_days=6000, seed=102)


@pytest.fixture
def trending():
    """One asset persistently trends above the other — the rebalancing bonus must be NEGATIVE."""
    return data.synthetic_trend(n_days=6000, seed=102)

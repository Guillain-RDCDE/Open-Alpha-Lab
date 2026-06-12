"""Shared fixtures for Study 84 (Moon-Math) — deterministic synthetic S2F tapes.

Tests never touch the network; all fixtures use the offline ``synthetic_s2f``
generator so the test-suite is reproducible and fast.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from moon_math import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where price is pure noise, S2F has no explanatory power."""
    df, truth = data.synthetic_s2f(n_days=1000, price_s2f_beta=0.0, seed=84)
    return df, truth


@pytest.fixture
def planb_tape():
    """A tape where price tracks S2F with PlanB's published slope (beta=3.3)."""
    df, truth = data.synthetic_s2f(n_days=2000, price_s2f_beta=3.3, seed=84)
    return df, truth


@pytest.fixture
def noisy_trend_tape():
    """A tape where price is a random walk (high vol, zero S2F beta) — the
    classic spurious regression scenario: two trending series, zero causal link."""
    df, truth = data.synthetic_s2f(n_days=2000, price_s2f_beta=0.0,
                                    price_vol=0.06, seed=42)
    return df, truth

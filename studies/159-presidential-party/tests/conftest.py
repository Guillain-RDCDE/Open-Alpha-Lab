"""Shared fixtures — deterministic synthetic monthly tapes with a known party premium.

Tests never touch the network and the only thing the party comparison can harvest
(a planted return premium under Democrats) is baked in or deliberately absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from presidential_party import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A synthetic tape with zero Democrat premium — the null hypothesis."""
    df, truth = data.synthetic_monthly(n_months=1200, dem_premium_bps=0.0, seed=159)
    return df, truth


@pytest.fixture
def planted_tape():
    """A synthetic tape with a very large planted Democrat premium (500 bps/month)
    so the effect is clearly recoverable against the ~460 bps/month noise floor."""
    df, truth = data.synthetic_monthly(n_months=1200, dem_premium_bps=500.0, seed=159)
    return df, truth

"""Shared fixtures for Study 230 (Ohlson O-score) tests.

All tests run on the synthetic panel -- deterministic, offline, no network,
no EDGAR dependency. The ``has_premium`` fixture plants a real O->return
relationship (positive o_premium = distressed firms earn more); the
``no_premium`` fixture is the null.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ohlson_o_score import data  # noqa: E402


@pytest.fixture
def has_premium():
    """Panel with a planted O-score -> return premium (o_premium = 0.08).

    Positive o_premium means high-O (distressed) firms earn more (risk-premium
    story). The hedge (safe - distressed) would then be *negative*.
    """
    o, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, o_premium=0.08, seed=230
    )
    return o, fwd, truth


@pytest.fixture
def no_premium():
    """Panel under the null: O-score has zero relationship with next-year returns."""
    o, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, o_premium=0.0, seed=230
    )
    return o, fwd, truth

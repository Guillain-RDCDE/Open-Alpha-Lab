"""Shared fixtures -- deterministic synthetic monthly panels with a *known* amount of
same-calendar-month seasonality, so tests never touch the network and the only thing
the top-decile portfolio can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from same_month_seasonality import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No same-month premium (signal = noise). Top decile should NOT beat bottom or random."""
    price_df, truth = data.synthetic_panel(
        n_firms=100, n_months=200, premium=0.0, seed=223
    )
    return price_df, truth


@pytest.fixture
def live_panel():
    """A planted +2% same-month premium. Top decile should outperform bottom decile."""
    price_df, truth = data.synthetic_panel(
        n_firms=100, n_months=200, premium=0.02, seed=223
    )
    return price_df, truth

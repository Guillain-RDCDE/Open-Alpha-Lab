"""Shared fixtures — deterministic synthetic cross-sectional panels with a *known*
fluency premium, so tests never touch the network and the signal can be verified as
present (premium > 0) or absent (premium = 0)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fluent_tickers import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel with zero fluency premium — the null hypothesis."""
    prices, mask, truth = data.synthetic_panel(
        n_stocks=40, n_fluent=20, n_days=2520, fluency_premium=0.0, seed=169
    )
    return prices, mask, truth


@pytest.fixture
def signal_panel():
    """A panel with a planted 10% annual fluency premium — the positive control.

    We use a large premium (10%/yr) over a long window (10y) so the planted
    effect reliably dominates sampling noise in the test suite.
    """
    prices, mask, truth = data.synthetic_panel(
        n_stocks=40, n_fluent=20, n_days=2520, fluency_premium=0.10, seed=169
    )
    return prices, mask, truth

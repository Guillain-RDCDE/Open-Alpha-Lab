"""Shared fixtures — deterministic synthetic annual cross-sections.

Tests never touch the network: the synthetic generator provides a fully reproducible
cross-section with a known planted effect (or none), so every assertion is verifiable
offline. The real-data path (load_real) is tested only for graceful failure when the
cache is absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from megacap_concentration import data  # noqa: E402


@pytest.fixture
def null_panel():
    """Synthetic cross-section with NO concentration effect (effect=0).

    Top-N should be statistically indistinguishable from the equal-weight baseline.
    """
    df, truth = data.synthetic_annual(
        n_years=18, n_stocks=100, top_n=10, effect=0.0, seed=177
    )
    return df, truth


@pytest.fixture
def signal_panel():
    """Synthetic cross-section with a strong planted concentration effect (effect=0.10).

    Top-N stocks get +10 ppt annual alpha — the concentration rule should clearly win.
    """
    df, truth = data.synthetic_annual(
        n_years=18, n_stocks=100, top_n=10, effect=0.10, seed=177
    )
    return df, truth

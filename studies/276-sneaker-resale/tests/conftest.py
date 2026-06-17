"""Shared fixtures for Study 276 (Sneaker-Resale).

All fixtures are deterministic and offline — they use either the hardcoded
sneaker-resale index or the synthetic generator, never the network or the
^GSPC parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sneaker_resale import data  # noqa: E402


@pytest.fixture
def sneaker_df():
    """The hardcoded sneaker-resale index (levels + year-over-year returns)."""
    return data.sneaker_index()


@pytest.fixture
def synthetic_null():
    """A synthetic panel with NO planted alpha and NO smoothing (the null)."""
    df, truth = data.synthetic_panel(n_years=18, alpha_bps=0.0, smooth_rho=0.0, seed=276)
    return df, truth


@pytest.fixture
def synthetic_smoothed():
    """A synthetic panel with heavy AR(1) smoothing but ZERO true alpha.

    This is the study's null-in-a-bottle for the stale-pricing illusion: no real
    outperformance, yet the smoothed Sharpe will look inflated.
    """
    df, truth = data.synthetic_panel(n_years=400, alpha_bps=0.0, smooth_rho=0.6, seed=276)
    return df, truth


@pytest.fixture
def synthetic_alpha():
    """A synthetic panel with a large planted sneaker alpha (positive control)."""
    df, truth = data.synthetic_panel(n_years=400, alpha_bps=2_000.0, smooth_rho=0.0, seed=276)
    return df, truth

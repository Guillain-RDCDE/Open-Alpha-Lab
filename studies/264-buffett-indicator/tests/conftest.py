"""Shared fixtures for Study 264 (Buffett-Indicator).

All fixtures are deterministic and offline -- they use either the hardcoded
Buffett-Indicator table or the synthetic generator, never the network or the
^GSPC cache (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buffett_indicator import data  # noqa: E402


@pytest.fixture
def bi_table():
    """The hardcoded Buffett-Indicator table (year-end, 1971-2025)."""
    return data.buffett_table()


@pytest.fixture
def synthetic_null():
    """A synthetic (valuation, fwd-return) frame with NO planted relationship."""
    df, truth = data.synthetic_panel(n_years=55, beta=0.0, seed=264)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic frame with a strong planted NEGATIVE (mean-reversion) effect."""
    df, truth = data.synthetic_panel(n_years=400, beta=-0.30, seed=264)
    return df, truth

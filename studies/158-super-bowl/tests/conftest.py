"""Shared fixtures for Study 158 (Super-Bowl).

All fixtures are deterministic and offline — they use either the hardcoded
Super Bowl table or the synthetic generator, never the network or the Shiller
parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from super_bowl import data  # noqa: E402


@pytest.fixture
def sb_table():
    """The hardcoded Super Bowl results table (all 59 games, 1967–2025)."""
    return data.superbowl_table()


@pytest.fixture
def synthetic_null():
    """A synthetic annual-return frame with NO planted effect (the null)."""
    df, truth = data.synthetic_annual(n_years=58, signal_bps=0.0, seed=158)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic annual-return frame with a strong planted NFC effect."""
    df, truth = data.synthetic_annual(n_years=58, signal_bps=2_000.0, seed=158)
    return df, truth

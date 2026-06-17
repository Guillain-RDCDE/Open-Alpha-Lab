"""Shared fixtures for Study 299 (Keynote-Drift).

All fixtures are deterministic and offline -- they use either the hardcoded
keynote table or the synthetic price generator, never the network or the AAPL
parquet cache (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from keynote_drift import data  # noqa: E402


@pytest.fixture
def keynotes():
    """The hardcoded Apple keynote table (2008-2025)."""
    return data.keynote_table()


@pytest.fixture
def synthetic_null():
    """A synthetic AAPL-like price frame with NO planted keynote drift (the null)."""
    df, truth = data.synthetic_prices(drift_bps=0.0, seed=299)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic AAPL-like price frame with a strong planted keynote drift."""
    df, truth = data.synthetic_prices(drift_bps=100.0, seed=299)
    return df, truth

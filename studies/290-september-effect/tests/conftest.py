"""Shared fixtures for Study 290 (September-Effect).

All fixtures are deterministic and offline -- they use the synthetic monthly
generator only, never the network or the Shiller parquet (so CI passes without
the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from september_effect import data  # noqa: E402


@pytest.fixture
def synthetic_null():
    """A synthetic monthly frame with NO planted September drag (the null)."""
    df, truth = data.synthetic_monthly(n_years=76, sep_bps=0.0, seed=290)
    return df, truth


@pytest.fixture
def synthetic_drag():
    """A synthetic monthly frame with a strong planted September drag."""
    df, truth = data.synthetic_monthly(n_years=76, sep_bps=-1000.0, seed=290)
    return df, truth

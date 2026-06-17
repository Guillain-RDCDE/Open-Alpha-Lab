"""Shared fixtures for Study 279 (Geomagnetic-Storms).

All fixtures are deterministic and offline — they use either the hardcoded
geomagnetic Ap index or the synthetic generator, never the network or the
^GSPC cache (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from geomagnetic_storms import data  # noqa: E402


@pytest.fixture
def kp_index():
    """The hardcoded monthly geomagnetic Ap index (1932–2025)."""
    return data.kp_monthly_index()


@pytest.fixture
def synthetic_null():
    """A synthetic monthly-return frame with NO planted storm penalty (the null)."""
    df, truth = data.synthetic_monthly(n_months=360, storm_bps=0.0, seed=279)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic monthly-return frame with a strong planted storm penalty."""
    df, truth = data.synthetic_monthly(n_months=600, storm_bps=200.0, seed=279)
    return df, truth

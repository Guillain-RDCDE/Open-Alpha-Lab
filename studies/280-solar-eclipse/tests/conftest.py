"""Shared fixtures for Study 280 (Solar-Eclipse).

All fixtures are deterministic and offline — they use either the hardcoded
solar-eclipse table or the synthetic generator, never the network or the ^GSPC
parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from solar_eclipse import data  # noqa: E402


@pytest.fixture
def ecl_table():
    """The hardcoded solar-eclipse table (central eclipses, 1928 onward)."""
    return data.eclipse_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily-return frame with NO planted eclipse effect (the null)."""
    df, truth = data.synthetic_panel(signal_bps=0.0, seed=280)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily-return frame with a strong planted eclipse drag."""
    df, truth = data.synthetic_panel(signal_bps=-300.0, seed=280)
    return df, truth

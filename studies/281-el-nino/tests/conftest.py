"""Shared fixtures for Study 281 (El-Nino).

All fixtures are deterministic and offline -- they use either the hardcoded NOAA
ONI table or the synthetic generator, never the network or the price cache (so CI
passes without the real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from el_nino import data  # noqa: E402


@pytest.fixture
def oni():
    """The hardcoded NOAA ONI (NDJ) table, all winters 1950-2024."""
    return data.oni_table()


@pytest.fixture
def synthetic_null():
    """A synthetic annual-return frame with NO planted ENSO effect (the null)."""
    df, truth = data.synthetic_panel(n_years=75, premium_bps=0.0, seed=281)
    return df.rename(columns={"ret": "asset_return"}), truth


@pytest.fixture
def synthetic_signal():
    """A synthetic annual-return frame with a strong planted El Nino premium."""
    df, truth = data.synthetic_panel(n_years=600, premium_bps=1_000.0, seed=281)
    return df.rename(columns={"ret": "asset_return"}), truth

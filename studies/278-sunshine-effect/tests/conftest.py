"""Shared fixtures for Study 278 (Sunshine-Effect).

All fixtures are deterministic and offline -- they use the hardcoded NYC cloud
climatology and the synthetic generator only, never the network or the ^GSPC
cache (so CI passes without any real-data cache present).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sunshine_effect import data  # noqa: E402


@pytest.fixture
def climatology():
    """The hardcoded NYC sky-cover climatology table (12 months)."""
    return data.cloud_climatology()


@pytest.fixture
def synthetic_null():
    """A synthetic (cloud, return) frame with NO planted effect (the null)."""
    df, truth = data.synthetic_cloud_returns(n_days=2520, effect_bps_per_octa=0.0, seed=278)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic frame with a strong planted sunshine effect."""
    df, truth = data.synthetic_cloud_returns(n_days=8800, effect_bps_per_octa=4.0, seed=278)
    return df, truth

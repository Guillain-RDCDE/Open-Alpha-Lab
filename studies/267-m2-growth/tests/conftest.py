"""Shared fixtures for Study 267 (M2-Growth).

All fixtures are deterministic and offline — they use either the hardcoded M2
YoY series or the synthetic generator, never the network or the ^GSPC cache (so
CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from m2_growth import data  # noqa: E402


@pytest.fixture
def m2_series():
    """The hardcoded monthly M2 YoY growth series (percent)."""
    return data.m2_yoy_series()


@pytest.fixture
def synthetic_null():
    """A synthetic (m2_yoy, sp_ret) panel with NO planted M2 link (the null)."""
    df, truth = data.synthetic_panel(n_months=360, premium=0.0, seed=267)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic panel with a strong planted M2 -> forward-return link."""
    df, truth = data.synthetic_panel(n_months=600, premium=2.0, seed=267)
    return df, truth

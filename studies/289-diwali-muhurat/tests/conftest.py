"""Shared fixtures for Study 289 (Diwali-Muhurat).

All fixtures are deterministic and offline — they use either the hardcoded
Diwali / Muhurat-session date table or the synthetic generator, never the
network or the INDA cache (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from diwali_muhurat import data  # noqa: E402


@pytest.fixture
def diwali_tbl():
    """The hardcoded Diwali / Muhurat-session date table (2003–2025)."""
    return data.diwali_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily-price frame with NO planted Diwali premium (the null)."""
    px, truth = data.synthetic_panel(premium_bps=0.0, seed=289)
    return px, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily-price frame with a strong planted Diwali premium."""
    px, truth = data.synthetic_panel(premium_bps=150.0, seed=289)
    return px, truth

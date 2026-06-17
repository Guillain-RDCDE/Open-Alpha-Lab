"""Shared fixtures for Study 284 (Equinox-Effect).

All fixtures are deterministic and offline — they use either the hardcoded
equinox/solstice table or the synthetic generator, never the network or the
^GSPC parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from equinox_effect import data  # noqa: E402


@pytest.fixture
def ev_table():
    """The hardcoded equinox/solstice table (1928–2026)."""
    return data.equinox_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily-return frame with NO planted event effect (the null)."""
    df, truth = data.synthetic_panel(signal_bps=0.0, seed=284)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily-return frame with a strong planted event drag."""
    df, truth = data.synthetic_panel(signal_bps=-300.0, seed=284)
    return df, truth

"""Shared fixtures for Study 297 (March-Madness).

All fixtures are deterministic and offline — they use either the hardcoded
NCAA tournament window table or the synthetic generator, never the network
(so CI passes without a real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from march_madness import data  # noqa: E402


@pytest.fixture
def tourney_table():
    """The hardcoded NCAA tournament window table (1985-2025, 2020 cancelled)."""
    return data.tournament_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily tape with NO planted tournament effect (the null)."""
    ret, mask, truth = data.synthetic_daily(
        n_years=20, madness_penalty_bp=0.0, seed=297
    )
    return ret, mask, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily tape with a strong planted tournament drag."""
    ret, mask, truth = data.synthetic_daily(
        n_years=20, madness_penalty_bp=-40.0, seed=297
    )
    return ret, mask, truth

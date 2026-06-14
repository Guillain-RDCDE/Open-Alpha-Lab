"""Shared fixtures — deterministic synthetic real-rate/equity worlds.

Tests are entirely offline and deterministic. The ``null_world`` fixture plants no
regime effect (the null hypothesis); the ``effect_world`` fixture plants a negative
regime effect so the engine must detect it to pass the positive-control test.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from real_rate_regime import data  # noqa: E402


@pytest.fixture
def null_world():
    """Synthetic world with zero regime effect — the timing rule is a coin here."""
    df, truth = data.synthetic_world(n_months=600, regime_effect=0.0, seed=119)
    return df, truth


@pytest.fixture
def effect_world():
    """Synthetic world with a strong planted regime effect (rising rates → lower returns)."""
    df, truth = data.synthetic_world(n_months=600, regime_effect=-2.0, seed=119)
    return df, truth

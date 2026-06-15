"""Shared fixtures — deterministic synthetic daily tapes with a *known* weekend boost,
so tests never touch the network and the planted signal can be confirmed absent or present.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_weekend import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk tape (weekend_boost=0) — no weekend anomaly planted."""
    bars, truth = data.synthetic_daily(n_years=8, weekend_boost=0.0, seed=175)
    return bars, truth


@pytest.fixture
def boosted_tape():
    """A tape with a substantial planted weekend boost (0.05 per day) — the engine must find it."""
    bars, truth = data.synthetic_daily(n_years=8, weekend_boost=0.05, seed=175)
    return bars, truth

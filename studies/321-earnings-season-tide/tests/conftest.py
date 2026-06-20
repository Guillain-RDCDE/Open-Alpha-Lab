"""Shared fixtures — deterministic synthetic daily tapes with a *known* earnings-season
tide (or none), so tests never touch the network and the only thing the calendar contrast
can detect (in-window drift) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from earnings_season_tide import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random walk (tide_bps=0) — the calendar contrast must read insignificant."""
    bars, truth = data.synthetic_daily(n_years=33, tide_bps=0.0, seed=321)
    return bars, truth


@pytest.fixture
def tide_tape():
    """A tape with a strong planted in-window tide (tide_bps=12) — must clear the bar."""
    bars, truth = data.synthetic_daily(n_years=33, tide_bps=12.0, seed=321)
    return bars, truth

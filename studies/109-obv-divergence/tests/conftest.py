"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
price momentum and volume lead structure, so tests never touch the network and the
only thing the OBV signals can harvest (price persistence or volume-as-lead-indicator)
is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from obv_divergence import data  # noqa: E402


@pytest.fixture
def null_tape():
    """Double null: no price momentum, no volume lead. Both OBV signals should behave like a coin."""
    bars, truth = data.synthetic_daily(n_days=500, price_momentum=0.0, vol_lead=0.0, seed=109)
    return bars, truth


@pytest.fixture
def trending_tape():
    """A tape with real price persistence (momentum 0.15) — the OBV trend signal might find it."""
    bars, truth = data.synthetic_daily(n_days=500, price_momentum=0.15, vol_lead=0.0, seed=109)
    return bars, truth


@pytest.fixture
def vol_lead_tape():
    """A tape where volume genuinely leads price (vol_lead 0.5) — the divergence signal should work."""
    bars, truth = data.synthetic_daily(n_days=500, price_momentum=0.0, vol_lead=0.5, seed=109)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
OpEx-week vol and return premium, so tests never touch the network and the planted
effect is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from monthly_opex import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no planted OpEx effect — range and volume identical on/off OpEx weeks."""
    bars, truth = data.synthetic_daily(
        n_years=20, vol_premium=0.0, ret_premium_bps=0.0, seed=195
    )
    return bars, truth


@pytest.fixture
def vol_tape():
    """A tape with a planted 40% range/volume uplift on OpEx weeks."""
    bars, truth = data.synthetic_daily(
        n_years=20, vol_premium=0.40, ret_premium_bps=0.0, seed=195
    )
    return bars, truth


@pytest.fixture
def ret_tape():
    """A tape with a planted +20 bps directional drift on OpEx-week days."""
    bars, truth = data.synthetic_daily(
        n_years=20, vol_premium=0.0, ret_premium_bps=20.0, seed=195
    )
    return bars, truth

"""Shared fixtures — deterministic synthetic annual tapes with a *known* January-Barometer
signal strength, so tests never touch the network and the only thing the barometer can
harvest (planted correlation between January sign and Feb-Dec return) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cold_open import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where January sign carries zero information about Feb-Dec (the null)."""
    df, truth = data.synthetic_annual(n_years=80, signal_strength=0.0, seed=80)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape where January sign genuinely predicts Feb-Dec direction (planted effect)."""
    df, truth = data.synthetic_annual(n_years=80, signal_strength=0.50, seed=80)
    return df, truth

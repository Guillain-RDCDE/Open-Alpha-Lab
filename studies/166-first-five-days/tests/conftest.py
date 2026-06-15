"""Shared fixtures — deterministic synthetic annual tapes with a *known* amount of
first-five-days signal, so tests never touch the network and the only thing the hit-rate
analysis can pick up is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from first_five_days import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no F5D signal (signal_strength=0) — hit-rate should be near the base rate."""
    df, truth = data.synthetic_annual(n_years=100, signal_strength=0.0, seed=166)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted F5D effect (signal_strength=0.5) — hit-rate should exceed base rate."""
    df, truth = data.synthetic_annual(n_years=100, signal_strength=0.5, seed=166)
    return df, truth

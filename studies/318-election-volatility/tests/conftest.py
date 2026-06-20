"""Shared fixtures — deterministic synthetic vol tapes with a *known* pre-election vol
spike and variance-risk premium, so tests never touch the network and the only thing the
event study can find (a planted spike / planted excess VRP) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from election_volatility import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No-signal tape (vol_bump=0, vrp=0): elections are ordinary days."""
    return data.synthetic_daily(vol_bump=0.0, vrp=0.0, seed=318)


@pytest.fixture
def spike_tape():
    """Strong planted pre-election vol spike and excess VRP (vol_bump=0.8, vrp=8)."""
    return data.synthetic_daily(vol_bump=0.8, vrp=8.0, seed=318)

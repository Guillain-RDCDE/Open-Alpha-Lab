"""Shared fixtures — deterministic synthetic crack/refiner tapes with a *known* lead-lag,
so tests never touch the network. The only thing a crack-timer acting on yesterday's crack
can harvest (a genuine predictive lead) is planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crack_spread import data  # noqa: E402


@pytest.fixture
def coincident():
    """A null tape (lead=0): the crack moves *with* the stock but cannot forecast it."""
    return data.synthetic_tape(n_days=4000, lead=0.0, seed=306)


@pytest.fixture
def leading():
    """A tape with a genuine predictive lead (lead=0.4) — the timer should win here."""
    return data.synthetic_tape(n_days=4000, lead=0.4, seed=306)

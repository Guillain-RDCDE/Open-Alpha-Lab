"""Shared fixtures — deterministic synthetic tapes with a *known* post-event drift,
so tests never touch the network and the only thing the event study can detect
(a planted rebound/decline around event dates) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bank_failure import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure-null tape (drift=0): event windows match random-date windows."""
    close, ev, truth = data.synthetic_tape(n_days=6000, drift=0.0, seed=316)
    return close, ev, truth


@pytest.fixture
def rebound_tape():
    """A strong planted post-event REBOUND (drift=+0.08): 'buy the blood' is true here."""
    close, ev, truth = data.synthetic_tape(n_days=6000, drift=0.08, seed=316)
    return close, ev, truth

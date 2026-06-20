"""Shared fixtures — deterministic synthetic Jackson-Hole tapes (a null and a planted
positive control), so tests never touch the network and the only thing the event study
can find (a speech-day bump) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from jackson_hole import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A null world (bump=0) — the event study must find nothing here."""
    return data.synthetic_world(n_years=33, bump=0.0, seed=314)


@pytest.fixture
def planted_tape():
    """A world with a strong planted speech-day bump (+80 bps) — the engine must detect it."""
    return data.synthetic_world(n_years=33, bump=0.008, seed=314)

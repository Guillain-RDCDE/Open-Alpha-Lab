"""Shared fixtures — deterministic synthetic inclusion event panels with known pop /
give-back magnitudes, so tests never touch the network and planted effects are either
present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from index_inclusion import data  # noqa: E402


@pytest.fixture
def null_events():
    """Synthetic panel with NO pop and NO give-back — the null hypothesis."""
    events, truth = data.synthetic_events(
        n_events=60, horizon=63, pop_bps=0.0, giveback_bps=0.0, seed=249
    )
    return events, truth


@pytest.fixture
def planted_events():
    """Synthetic panel with a PLANTED pop (pop_bps=20) — positive control."""
    events, truth = data.synthetic_events(
        n_events=60, horizon=63, pop_bps=20.0, giveback_bps=0.0, seed=249
    )
    return events, truth


@pytest.fixture
def giveback_events():
    """Synthetic panel with a PLANTED give-back (giveback_bps=15) — negative control."""
    events, truth = data.synthetic_events(
        n_events=60, horizon=63, pop_bps=0.0, giveback_bps=15.0, seed=249
    )
    return events, truth

"""Shared fixtures — deterministic synthetic SPY-like tapes with a *known* (planted or
absent) downgrade dip around the events, so tests never touch the network and the only
thing the event-study engine can find is what we baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sovereign_downgrade import data  # noqa: E402


@pytest.fixture
def planted():
    """A tape with a real downgrade dip (effect>0) — the engine should find it."""
    return data.synthetic_spy(event_effect=0.04, n_events=40, seed=315)


@pytest.fixture
def null():
    """A tape with no planted dip — the engine should find nothing (clean seed)."""
    return data.synthetic_spy(event_effect=0.0, n_events=40, seed=317)

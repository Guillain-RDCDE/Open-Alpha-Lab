"""Shared fixtures — deterministic synthetic VIX-like tapes with a *known* (planted or
absent) vol hump around the events, so tests never touch the network and the only thing
the event-study engine can find is what we baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from debt_ceiling import data  # noqa: E402


@pytest.fixture
def planted():
    """A tape with a real vol hump (effect>0) — the engine should find it."""
    return data.synthetic_vix(event_effect=0.5, pre=20, post=20, n_events=40, seed=312)


@pytest.fixture
def null():
    """A tape with no planted hump — the engine should find nothing (clean seed)."""
    return data.synthetic_vix(event_effect=0.0, pre=20, post=20, n_events=40, seed=314)

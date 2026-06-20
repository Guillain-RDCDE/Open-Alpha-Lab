"""Shared fixtures — deterministic synthetic daily tapes with a *known* (planted or
absent) post-event bounce, so tests never touch the network and the only thing the
event-study engine can find is what we baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from government_shutdown import data  # noqa: E402


@pytest.fixture
def planted():
    """A tape with a real post-event bounce (effect>0) — the engine should find it."""
    return data.synthetic_daily(event_effect=0.002, event_window=20, n_events=40, seed=311)


@pytest.fixture
def null():
    """A tape with no planted effect — the engine should find nothing (clean seed)."""
    return data.synthetic_daily(event_effect=0.0, event_window=20, n_events=40, seed=314)

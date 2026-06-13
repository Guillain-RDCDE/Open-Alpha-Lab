"""Shared fixtures — deterministic synthetic session panels with a known amount of
last-hour continuation, so tests never touch the network and the only thing the
momentum signal can harvest (morning→last-hour prediction) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from power_hour import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel with zero continuation — the last hour is independent of the morning."""
    panel, truth = data.synthetic_sessions(n_sessions=500, continuation=0.0, seed=116)
    return panel, truth


@pytest.fixture
def trending_panel():
    """A panel with real continuation (0.30) — the follow-morning signal should win here."""
    panel, truth = data.synthetic_sessions(n_sessions=500, continuation=0.30, seed=116)
    return panel, truth


@pytest.fixture
def reversal_panel():
    """A panel with strong reversal (−0.30) — the fade signal should win here."""
    panel, truth = data.synthetic_sessions(n_sessions=500, continuation=-0.30, seed=116)
    return panel, truth

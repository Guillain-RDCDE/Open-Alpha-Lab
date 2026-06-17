"""Shared fixtures — deterministic synthetic reverse-split event panels with a *known*
amount of post-RS drift (positive or negative), so tests never touch the network and
the planted effect is either present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reverse_split import data  # noqa: E402


@pytest.fixture
def null_events():
    """A synthetic panel with NO post-RS drift (drift_bps=0) — the null hypothesis."""
    events, truth = data.synthetic_events(
        n_stocks=10, n_events=40, horizon=252, drift_bps=0.0, seed=250
    )
    return events, truth


@pytest.fixture
def planted_events():
    """A synthetic panel with a PLANTED negative drift (drift_bps=-20) — positive control."""
    events, truth = data.synthetic_events(
        n_stocks=10, n_events=40, horizon=252, drift_bps=-20.0, seed=250
    )
    return events, truth

"""Shared fixtures — deterministic synthetic split event panels with a *known* amount
of post-split drift, so tests never touch the network and the planted effect is either
present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from split_drift import data  # noqa: E402


@pytest.fixture
def null_events():
    """A synthetic panel with NO post-split drift (drift_bps=0) — the null hypothesis."""
    events, truth = data.synthetic_events(
        n_stocks=10, n_events=40, horizon=252, drift_bps=0.0, seed=142
    )
    return events, truth


@pytest.fixture
def planted_events():
    """A synthetic panel with a PLANTED post-split drift (drift_bps=20) — the positive control."""
    events, truth = data.synthetic_events(
        n_stocks=10, n_events=40, horizon=252, drift_bps=20.0, seed=142
    )
    return events, truth

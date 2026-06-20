"""Shared fixtures — deterministic synthetic event-time panels with a *known* lock-up sag
(or a deliberate null), so tests never touch the network and the only thing the event-study
engine can detect (an abnormal sag on the expiry bar) is baked in or absent on purpose."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lockup_expiry import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No planted sag (sag_bps=0) — the engine must NOT find a significant negative CAR."""
    panel, truth = data.synthetic_panel(n_events=60, sag_bps=0.0, pre_drift_bps=0.0, seed=319)
    return panel, truth


@pytest.fixture
def sag_panel():
    """A strong planted one-shot sag on the expiry bar — the engine should detect it."""
    panel, truth = data.synthetic_panel(n_events=60, sag_bps=-300.0, pre_drift_bps=0.0, seed=319)
    return panel, truth

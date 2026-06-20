"""Shared fixtures — deterministic synthetic SPY-like tapes with a *known* post-shock
drift, so the tests never touch the network and the only thing the event study can find
(a planted abnormal-return path) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from geopolitical_shock import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A no-signal tape (shock_drift=0) — events carry no information."""
    return data.synthetic_daily(shock_drift=0.0, seed=313)


@pytest.fixture
def relief_tape():
    """A tape with a strong planted post-shock relief rally (shock_drift=0.03)."""
    return data.synthetic_daily(shock_drift=0.03, seed=313)

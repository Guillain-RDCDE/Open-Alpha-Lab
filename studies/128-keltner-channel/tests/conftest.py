"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
momentum / mean-reversion baked in, so tests never touch the network and both the
breakout and reversion arms can be validated against a planted effect."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from keltner_channel import data  # noqa: E402


@pytest.fixture
def flat():
    """A martingale tape (momentum 0) — both channel rules must behave like fair coins."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=128)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real return persistence (momentum +0.20) — breakout arm should win."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.20, seed=128)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with planted mean-reversion (momentum -0.20) — reversion arm should win."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=-0.20, seed=128)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
SKEW predictive power, so tests never touch the network and the study's thesis
(high SKEW predicts lower forward returns) is verifiable against both a null tape
and a tape where the signal is planted."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tail_radar import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where SKEW carries zero predictive power — the null in a bottle."""
    df, truth = data.synthetic_daily(n_days=2000, skew_signal=0.0, seed=86)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with real planted SKEW signal — high SKEW predicts lower SPY returns."""
    df, truth = data.synthetic_daily(n_days=2000, skew_signal=0.015, seed=86)
    return df, truth

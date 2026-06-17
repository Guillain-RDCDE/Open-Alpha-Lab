"""Shared fixtures -- deterministic synthetic (mood, return) panels with a *known*
amount of lead-lag predictability, so tests never touch the network and the only
thing the strategy can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from twitter_mood import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No lead-lag (mood = noise). The slope should be statistically zero."""
    df, truth = data.synthetic_series(n_days=200, predictive=0.0, seed=256)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted lead-lag (predictive>0). Lagged Calm should predict the return."""
    df, truth = data.synthetic_series(n_days=200, predictive=0.004, seed=256)
    return df, truth

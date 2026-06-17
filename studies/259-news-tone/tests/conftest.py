"""Shared fixtures -- deterministic synthetic daily (tone, next-day return) panels
with a *known* amount of next-day predictability, so tests never touch the network
and the only thing the tone-timing strategy can harvest is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from news_tone import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No next-day predictability (gamma=0). Tone is a useless AR(1) series."""
    df, truth = data.synthetic_daily(n_days=2000, gamma=0.0, seed=259)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted next-day link (gamma=0.003). Tone forecasts tomorrow's return."""
    df, truth = data.synthetic_daily(n_days=2000, gamma=0.003, seed=259)
    return df, truth

"""Shared fixtures — deterministic synthetic tapes for Study 912 (Gold + Trend).

Two fixtures, both offline and deterministic (fixed seed 912; no network):

- ``dead_decade`` — a two-state tape with a persistent, low-vol *grind-down* dead decade
  (``signal_strength=1``): a genuine, trend-followable regime the 200-day filter can duck.
- ``flat_vol`` — a single-regime tape (``signal_strength=0``): the trend filter has nothing
  to read, so any excess-Sharpe advantage it shows is noise (the null).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_trend import data  # noqa: E402


@pytest.fixture
def dead_decade():
    """A persistent dead-decade regime the 200-day trend filter can genuinely duck."""
    prices, truth = data.synthetic_daily(signal_strength=1.0, seed=912)
    return prices, truth


@pytest.fixture
def flat_vol():
    """A single-regime null tape — no trend signal to read (the null control)."""
    prices, truth = data.synthetic_daily(signal_strength=0.0, seed=912)
    return prices, truth

"""Shared fixtures — deterministic synthetic tapes for Study 948 (Capture Ratio).

Three fixtures, all offline and deterministic (fixed seed 948; no network):

- ``planted_panel`` — a monthly cross-section with a *known* capture structure
  (``signal_strength=1``): one genuinely convex fund, three genuinely concave ones, three
  flat. The scorecard must recover the planted spreads and rank the funds correctly.
- ``null_panel`` — the same cross-section with every convexity switched off
  (``signal_strength=0``): every fund is plain linear beta, so every *true* capture spread
  is exactly zero. Nothing should fire beyond the nominal rate.
- ``daily_tape`` — a single fund/benchmark/cash trio as daily price levels, so the
  daily → monthly resampling path is exercised end to end.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from capture_ratio import data  # noqa: E402


@pytest.fixture
def planted_panel():
    """A cross-section with a known, non-zero capture structure."""
    return data.synthetic_panel(signal_strength=1.0, seed=948)


@pytest.fixture
def null_panel():
    """The null: every fund is plain linear beta, true capture spread exactly zero."""
    return data.synthetic_panel(signal_strength=0.0, seed=948)


@pytest.fixture
def daily_tape():
    """A daily fund/benchmark/cash price tape with a planted convex payoff."""
    return data.synthetic_daily(signal_strength=1.0, seed=948)

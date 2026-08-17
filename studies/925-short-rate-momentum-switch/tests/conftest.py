"""Shared fixtures — deterministic synthetic tapes for Study 925 (Front-End Trend).

Both fixtures are offline and deterministic (fixed seed 925; no network, no cache):

- ``trending_rates`` — a world where the short rate's daily *increments* are strongly
  autocorrelated (``signal_strength=1``): the front end genuinely trends, so a trailing
  3-month rate change forecasts the next move and the switch rule should win.
- ``random_walk_rates`` — i.i.d. increments (``signal_strength=0``): the front end is a
  pure random walk, the trailing change is noise, and the switch rule must **not** beat
  the static intermediate-duration leg (the null).

Unconditional rate volatility is held fixed across the knob, so the two worlds differ
only in predictability, not in risk.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from front_end_trend import data  # noqa: E402


@pytest.fixture
def trending_rates():
    """A front end that genuinely trends — the switch rule should find it."""
    return data.synthetic_daily(signal_strength=1.0, seed=925)


@pytest.fixture
def random_walk_rates():
    """A front end with i.i.d. increments — the null; nothing to trend-follow."""
    return data.synthetic_daily(signal_strength=0.0, seed=925)

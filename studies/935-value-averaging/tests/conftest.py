"""Shared fixtures — deterministic synthetic tapes for Study 935 (Value Averaging).

Three fixtures, all offline and deterministic (fixed seeds; no network, no cache):

- ``wobbly`` — a tape carrying a large transitory, mean-reverting swing
  (``signal_strength=1``): the world value averaging is *designed* for, where its
  contrarian schedule must pay.
- ``random_walk`` — the same generator with the swing switched off
  (``signal_strength=0``): a pure drifting random walk with nothing to lean against.
- ``deterministic`` — a zero-volatility constant-growth tape: the hard null, where
  VA and an exposure-matched DCA must produce essentially the same wealth.

Twelve-year tapes keep the suite fast; the rolling race still yields ~100 windows.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from value_avg import data  # noqa: E402


@pytest.fixture(scope="session")
def wobbly():
    """A strongly mean-reverting tape — the planted effect value averaging harvests."""
    return data.synthetic_daily(n_years=12, signal_strength=1.0, seed=935)


@pytest.fixture(scope="session")
def random_walk():
    """A drifting random walk — no predictability for the contrarian rule to find."""
    return data.synthetic_daily(n_years=12, signal_strength=0.0, seed=935)


@pytest.fixture(scope="session")
def deterministic():
    """A zero-vol constant-growth tape — the hard null (nothing to trade against)."""
    return data.synthetic_daily(n_years=12, signal_strength=0.0, seed=935, vol_ann=0.0)

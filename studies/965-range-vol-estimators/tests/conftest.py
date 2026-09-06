"""Shared fixtures for Study 965 — deterministic, offline, no network.

The planted world is a simulated intraday path with a *known* daily sigma, which is
the only place an efficiency claim can honestly be tested — on real data there is no truth to
compare against. The null world holds volatility constant, so an estimator's dispersion there
is pure estimation error and nothing else.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from range_vol import data  # noqa: E402

N_YEARS = 12


@pytest.fixture
def planted():
    """Bars with clustered volatility (signal_strength=1) — the interesting world."""
    return data.synthetic_ohlc(n_years=N_YEARS, signal_strength=1.0, seed=965)


@pytest.fixture
def null_bars():
    """The matching null: the same machinery with volatility held constant."""
    return data.synthetic_ohlc(n_years=N_YEARS, signal_strength=0.0, seed=965)


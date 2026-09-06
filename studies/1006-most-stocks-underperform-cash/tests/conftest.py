"""Shared fixtures for Study 1006 — deterministic, offline, no network.

The synthetic world draws stocks from a lognormal distribution with a
**tunable** volatility and a common drift. That is the whole mechanism in one knob: with zero
volatility the median equals the mean; raising volatility pushes the median below the mean by
exactly the variance drag, with no change to the expected return at all. Any claim the study
makes about skewness causing the gap must reproduce in that world and vanish when the knob is
turned off.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from moststocks import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=1006)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=1006)


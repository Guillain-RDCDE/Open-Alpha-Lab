"""Shared fixtures for Study 1007 — deterministic, offline, no network.

The synthetic world generates returns that are **i.i.d. by construction**, so
mean reversion is impossible. Any narrowing of annualised dispersion there is pure arithmetic —
the √T in the denominator — and cannot be evidence for time diversification. That separation is
the study's central control: it distinguishes the arithmetic effect, which is real and
uninteresting, from genuine mean reversion, which would be interesting and has to be tested for
separately.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from timediv import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=1007)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=1007)


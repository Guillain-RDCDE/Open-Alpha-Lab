"""Shared fixtures for Study 1025 — deterministic, offline, no network.

The synthetic world is i.i.d. lognormal stocks, bonds and bills with a **planted** equity
premium. Because the generating distribution is known, the break-even horizon a loss-averse
investor should find is known too — it exists at ``signal_strength = 1`` and does not exist at
``signal_strength = 0`` (stocks then earn no more than bills on average). Both worlds come from
the same generator with one knob moved: a detector that fires on both is not detecting anything.

The long worlds (6,000 months) are for checking the machinery; the 92-year worlds show how
noisy the same calculation is on a sample the length of the real tape.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dontlook import data  # noqa: E402

LONG = 6000


@pytest.fixture(scope="session")
def planted():
    """500 years of i.i.d. monthly returns with the planted premium (signal_strength=1)."""
    return data.synthetic_monthly(n_months=LONG, signal_strength=1.0, seed=1025)[0]


@pytest.fixture(scope="session")
def null_world():
    """The matching null: stocks earn the bill rate on average (signal_strength=0)."""
    return data.synthetic_monthly(n_months=LONG, signal_strength=0.0, seed=1025)[0]


@pytest.fixture(scope="session")
def planted_daily():
    return data.synthetic_daily(n_years=30, signal_strength=1.0, seed=1025)[0]

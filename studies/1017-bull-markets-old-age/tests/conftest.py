"""Shared fixtures for Study 1017 — deterministic, offline, no network.

The synthetic world is a monthly market whose bulls either **genuinely age** (``planted``,
``signal_strength=1``: past three years, each month carries a rising, planted probability of a
bull-ending crash) or are the **exact random-walk null** (``null_world``, ``signal_strength=0``).
Both come from the same generator with one knob moved, which is the whole point: a detector that
fires on the planted world *and* on the null is detecting the dating rule, not ageing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oldage import data  # noqa: E402

N_MONTHS = 1109     # the length of the real FF tape


@pytest.fixture(scope="session")
def planted():
    """Bulls that really do die of old age (signal_strength=1)."""
    return data.synthetic_monthly(n_months=N_MONTHS, signal_strength=1.0, seed=1017)


@pytest.fixture(scope="session")
def null_world():
    """The matched null: an i.i.d. lognormal random walk (signal_strength=0)."""
    return data.synthetic_monthly(n_months=N_MONTHS, signal_strength=0.0, seed=1017)

"""Shared fixtures for Study 968 — deterministic, offline, no network.

The two fixtures here are the two worlds the bootstrap has to survive: a planted
world with volatility clustering and autocorrelation (where an i.i.d. resample destroys exactly
the structure that matters) and the i.i.d. null (where every method should agree and any that
does not is broken).

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from boot_choice import data  # noqa: E402

N_YEARS = 8


@pytest.fixture
def planted():
    """Dependent returns: AR(1) plus volatility clustering plus fat tails."""
    return data.synthetic_returns(n_years=N_YEARS, ar1=0.15, signal_strength=1.0, seed=968)


@pytest.fixture
def iid_returns():
    """The i.i.d. null — every bootstrap must deliver nominal coverage here."""
    return data.synthetic_returns(n_years=N_YEARS, ar1=0.0, signal_strength=0.0, seed=968)


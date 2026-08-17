"""Shared fixtures — deterministic synthetic panels for Study 962 (Do It Yourself).

Both fixtures are offline, deterministic (fixed seed 962) and network-free:

- ``fee_panel`` — a one-factor sector world in which the fund charges a real annual fee
  (``signal_strength=1``): the do-it-yourself basket of top holdings should recover
  roughly that fee, with an HAC *t* far above 2.
- ``null_panel`` — the same world with the fee switched off (``signal_strength=0``): the
  basket must show no reliable gap at all (the null).

They are ``session``-scoped because building a 20-year, 50-name panel and walking the
basket day by day is the slowest thing in the suite; the fixtures are read-only.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from diy_sector import data  # noqa: E402


@pytest.fixture(scope="session")
def fee_panel():
    """A fund that skims a real fee — the planted effect the detector must recover."""
    return data.synthetic_panel(signal_strength=1.0, seed=962)


@pytest.fixture(scope="session")
def null_panel():
    """The same world with no fee at all — the detector must stay quiet."""
    return data.synthetic_panel(signal_strength=0.0, seed=962)


@pytest.fixture(scope="session")
def small_panel():
    """A short panel for the fast mechanical tests (shape, costs, lag, turnover)."""
    return data.synthetic_panel(n_years=4, n_names=6, n_tail=10, seed=962)

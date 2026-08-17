"""Shared fixtures — deterministic synthetic tapes for Study 944 (How Much Leverage).

Two fixtures, both offline and deterministic (fixed seed 944; no network, no cache):

- ``planted`` — an i.i.d. tape whose growth-optimal leverage is **2.0** by construction
  (``signal_strength=1``): the sweep must find a peak there, on average across seeds.
- ``null`` — the same tape with **zero** excess drift (``signal_strength=0``): every unit
  of leverage is pure variance drag, so the optimum must sit on the grid floor.

The synthetic grid runs from 0.0 (unlike the real-tape grid, which starts at 1.0) so the
null has somewhere to go — otherwise "the optimum is at the floor" would be untestable.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from optimal_leverage import data  # noqa: E402

SYN_GRID = np.round(np.arange(0.0, 3.0001, 0.25), 4)


@pytest.fixture
def planted():
    """A 40-year i.i.d. tape with a planted growth-optimal leverage of 2.0."""
    return data.synthetic_daily(signal_strength=1.0, seed=944)


@pytest.fixture
def null():
    """The same tape with zero excess drift — leverage is pure drag (the null)."""
    return data.synthetic_daily(signal_strength=0.0, seed=944)

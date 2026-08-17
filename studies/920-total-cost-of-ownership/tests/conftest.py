"""Shared fixtures — deterministic synthetic wrapper pairs for Study 920.

Every fixture is offline and deterministic (fixed seed 920; no network, no cache):

- ``planted_pair`` — two wrappers on one index separated by a planted 6 bp/yr drag gap,
  the size of a real fee difference between a 1993 trust and its modern clone.
- ``null_pair`` — the same two wrappers with **no** gap (``signal_strength=0``): identical
  funds up to closing-print noise and a slow level wobble. Any tracking difference the
  estimator reports here is spurious.
- ``panel`` — a ladder of planted gaps (0, 3, 6, 12 bp/yr) for the calibration check.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tco import data  # noqa: E402


@pytest.fixture
def planted_pair():
    """A wrapper pair with a genuine 6 bp/yr drag gap the estimator must recover."""
    return data.synthetic_daily(signal_strength=1.0, gap_bp_yr=6.0, seed=920)


@pytest.fixture
def null_pair():
    """Two identical wrappers — the null; the estimator must stay quiet."""
    return data.synthetic_daily(signal_strength=0.0, gap_bp_yr=6.0, seed=920)


@pytest.fixture
def panel():
    """A ladder of planted gaps for the calibration check."""
    return data.synthetic_panel(gaps_bp_yr=(0.0, 3.0, 6.0, 12.0), seed=920)

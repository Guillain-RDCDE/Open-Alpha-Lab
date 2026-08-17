"""Shared fixtures — deterministic synthetic front-end panels for Study 922.

Two fixtures, both offline and deterministic (fixed seed 922; no network):

- ``planted`` — a front-end panel in which the fixed leg carries a real 1.85-year
  duration (``signal_strength=1``), so the direction of the short rate genuinely flips
  the ranking of the floater and the fixed note.
- ``null`` — the *same* rate cycle, but the fixed leg has no duration at all
  (``signal_strength=0``). Rates still rise and fall exactly as much, so the regime
  classifier is just as busy; there is simply nothing for it to find. Any contrast the
  estimator reports here is noise.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from frn_front import data  # noqa: E402


@pytest.fixture
def planted():
    """A front-end panel with a real 1.85-year duration on the fixed leg."""
    return data.synthetic_panel(signal_strength=1.0, seed=922)


@pytest.fixture
def null():
    """The same rate cycle with a zero-duration 'fixed' leg — the null."""
    return data.synthetic_panel(signal_strength=0.0, seed=922)

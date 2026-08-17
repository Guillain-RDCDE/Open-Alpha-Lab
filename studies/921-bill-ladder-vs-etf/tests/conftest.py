"""Shared fixtures — deterministic synthetic tapes for Study 921 (Bill Ladder vs ETF).

Two fixtures, both offline and deterministic (fixed seed 921; no network):

- ``fee_world`` — a mean-reverting short-rate path plus a cash ETF that charges a *known*
  13.5 bps expense ratio (``signal_strength=1``). A correctly built ladder must recover a
  gap of roughly that size.
- ``free_world`` — the same rate path with a *free* ETF (``signal_strength=0``): the null.
  A correctly built ladder must show no gap at all.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bill_ladder import data  # noqa: E402


@pytest.fixture
def fee_world():
    """A synthetic cash ETF charging a planted 13.5 bps fee — the effect to recover."""
    frame, truth = data.synthetic_daily(signal_strength=1.0, seed=921)
    return frame, truth


@pytest.fixture
def free_world():
    """A synthetic cash ETF charging nothing — the null, where the gap must vanish."""
    frame, truth = data.synthetic_daily(signal_strength=0.0, seed=921)
    return frame, truth

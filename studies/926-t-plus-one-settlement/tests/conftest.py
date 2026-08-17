"""Shared fixtures — deterministic synthetic panels for Study 926 (T+1).

Two fixtures, both offline and deterministic (fixed seed 926; no network):

- ``planted`` — a two-asset panel in which the treated leg's overnight drift, overnight
  volatility and turn-of-month overnight drift all shift at the planted switch date
  (``signal_strength=1``). The difference-in-difference estimator must recover it.
- ``null_world`` — the same panel with nothing changing at the switch
  (``signal_strength=0``). The estimator must stay quiet.

No test in this suite reads ``studies/_cache``; the whole suite runs on a fresh checkout
with no cache present.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from t_plus_one import data  # noqa: E402


@pytest.fixture
def planted():
    """A panel with a genuine, known structural break in the treated leg."""
    return data.synthetic_panel(signal_strength=1.0, seed=926)


@pytest.fixture
def null_world():
    """A panel in which the switch date changes nothing at all (the null)."""
    return data.synthetic_panel(signal_strength=0.0, seed=926)

"""Shared fixtures — deterministic synthetic tapes for Study 938 (Open or Close).

Three fixtures, all offline and deterministic (fixed seed 938; no network):

- ``planted`` — a tape whose daily move is deliberately booked open-to-close after a strong
  or weak month (``signal_strength=1``): filling a trend rule at the **open** must win.
- ``null`` — the same close-to-close tape with the intraday/overnight split randomised
  (``signal_strength=0``): the two execution venues must tie.
- ``panel_null`` — four independent null worlds, for the pooled-book path.

Because the generator splits a *fixed* close-to-close path, the close-executed arm is
bit-identical across ``signal_strength`` — the knob can only move the open arm.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from open_close_exec import data  # noqa: E402


@pytest.fixture(scope="session")
def planted():
    """Planted intraday continuation — the open fill should capture it."""
    return data.synthetic_daily(signal_strength=1.0, seed=938)


@pytest.fixture(scope="session")
def null():
    """No continuation — the venue choice must be worth nothing."""
    return data.synthetic_daily(signal_strength=0.0, seed=938)


@pytest.fixture(scope="session")
def panel_null():
    """Four independent null worlds (the pooled-book path, offline)."""
    return data.synthetic_panel(n_assets=4, signal_strength=0.0, seed=938)

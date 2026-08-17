"""Shared fixtures — deterministic synthetic tapes for Study 919 (Methodology Shock).

Every fixture is offline and deterministic (fixed seed 919; no network, no cache):

- ``planted_pair`` / ``null_pair`` — a single treated/control price pair with, or without,
  a planted post-announcement abnormal drift.
- ``planted_panel`` / ``null_panel`` — three independent pairs, the synthetic mirror of
  the real study's QQQ/SPY, SPY/IWM and IWM/MDY structure.

The planted world is what the detector must recover; the null world is what it must stay
quiet on. Neither ever supports a stamp on the real tape.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from methodology_shock import data  # noqa: E402


@pytest.fixture
def planted_pair():
    """One pair carrying a genuine +250 bps post-announcement methodology shock."""
    return data.synthetic_daily(signal_strength=1.0, seed=919)


@pytest.fixture
def null_pair():
    """One pair with nothing planted — the null the detector must not fire on."""
    return data.synthetic_daily(signal_strength=0.0, seed=919)


@pytest.fixture
def planted_panel():
    """Three independent pairs, each with planted shocks (the powered world)."""
    return data.synthetic_panel(signal_strength=1.0, seed=919)


@pytest.fixture
def null_panel():
    """Three independent pairs with nothing planted (the null world)."""
    return data.synthetic_panel(signal_strength=0.0, seed=919)

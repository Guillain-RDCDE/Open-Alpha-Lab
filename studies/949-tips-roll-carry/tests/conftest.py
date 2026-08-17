"""Shared fixtures — deterministic synthetic tapes for Study 949 (Riding the TIPS Curve).

Every fixture is offline and deterministic (fixed seed 949; no network, no cache):

- ``planted`` — one linker / nominal / cash tape carrying a genuine 2%/yr residual
  carry the duration hedge should recover (``signal_strength=1``).
- ``null`` — the same world with **no** residual carry at all
  (``signal_strength=0``); the hedged alpha must be indistinguishable from zero.
- ``planted_panel`` / ``null_panel`` — the three-bucket ladder versions, used to check
  that the ladder race does not confuse duration with carry.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tips_roll import data  # noqa: E402


@pytest.fixture
def planted():
    """A tape with a real, duration-hedgeable residual carry of 2%/yr."""
    return data.synthetic_daily(signal_strength=1.0, seed=949)


@pytest.fixture
def null():
    """The null: identical duration factor, zero residual carry."""
    return data.synthetic_daily(signal_strength=0.0, seed=949)


@pytest.fixture
def planted_panel():
    """Three maturity buckets, the same carry planted in each."""
    return data.synthetic_panel(signal_strength=1.0, seed=949)


@pytest.fixture
def null_panel():
    """Three maturity buckets, no carry planted anywhere."""
    return data.synthetic_panel(signal_strength=0.0, seed=949)

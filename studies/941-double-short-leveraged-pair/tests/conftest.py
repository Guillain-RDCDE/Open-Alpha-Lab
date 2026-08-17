"""Shared fixtures — deterministic synthetic tapes for Study 941 (Short Both Legs).

Two fixtures, both offline and deterministic (fixed seed 941; no network). Both worlds
carry the *same*, large volatility decay in their levered legs; the only difference is
whether the funds charge a fee:

- ``costly_funds`` — the funds charge a realistic all-in load (``signal_strength=1``);
  an equal-dollar short of both legs must recover the average of the two loads.
- ``free_funds`` — cost-free funds (``signal_strength=0``); the decay is untouched, so a
  harness that "harvests decay" would still fire here. It must not: the harvest has to
  collapse to zero (the null).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_pair import data  # noqa: E402


@pytest.fixture
def costly_funds():
    """High-vol world where both levered funds charge a realistic fee load."""
    prices, truth = data.synthetic_daily(signal_strength=1.0, seed=941)
    return prices, truth


@pytest.fixture
def free_funds():
    """The null: identical volatility decay, but the funds are free — no harvest exists."""
    prices, truth = data.synthetic_daily(signal_strength=0.0, seed=941)
    return prices, truth

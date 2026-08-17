"""Shared fixtures — deterministic synthetic panels for Study 913.

Two fixtures, both offline and deterministic (fixed seed 913; no network):

- ``fee_ladder`` — a panel of four index funds separated by a 30 bp top-to-bottom fee
  ladder against an 8 bp annual tracking-noise floor (``signal_strength=1``). The ladder is
  wide enough to show through, so the tracking-difference rank genuinely persists.
- ``flat_fees`` — the same panel with every fund charging the identical fee
  (``signal_strength=0``). Annual tracking differences are pure noise, so the rank must not
  persist and no rule may out-earn another (the null).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from td_persist import data  # noqa: E402


@pytest.fixture
def fee_ladder():
    """A panel with a genuine, persistent fee ladder the pipeline should detect."""
    prices, truth = data.synthetic_panel(signal_strength=1.0, seed=913)
    return prices, truth


@pytest.fixture
def flat_fees():
    """A panel with identical fees — annual tracking-difference ranks are pure noise."""
    prices, truth = data.synthetic_panel(signal_strength=0.0, seed=913)
    return prices, truth
